"""Daily and total loss limits as a persisted state machine.

States, in order of precedence:

* ``HARD_STOPPED``: the total loss reached the book's total limit, measured from the baseline
  (starting capital, or the equity at the last approved recovery reset). Latched in the store
  until ``reset_baseline``; recovering equity does not lift it.
* ``DAILY_PAUSED``: the loss since the previous trading day's closing snapshot reached the daily
  limit. Latched for that IST calendar day only.
* ``ACTIVE``.

Everything that matters is persisted, so a restart sees the same state and ``newly_triggered``
fires once per transition across restarts, not once per process. Store keys, per book:
``baseline:<book>`` (float), ``paused_on:<book>`` (ISO date, or null), ``hard_stop:<book>``
(bool) and ``stop_id:<book>`` (int, bumped by every ``reset_baseline``).

Cash is stored unrounded, so a loss of "exactly" the limit can arrive as 99.9999999. A limit
counts as reached from ``limit - TOLERANCE`` (a millionth of a rupee) so float noise can never
let a breach slip through, and a genuine 99.99 against a 100 limit still does not trigger.
"""

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from aitrader.calendar import IST, previous_trading_day
from aitrader.config import BookConfig
from aitrader.store import Store

log = logging.getLogger(__name__)

TOLERANCE = 1e-6


class LimitState(StrEnum):
    ACTIVE = "ACTIVE"
    DAILY_PAUSED = "DAILY_PAUSED"
    HARD_STOPPED = "HARD_STOPPED"


@dataclass(frozen=True)
class LimitEval:
    """Result of one evaluation. Losses are positive when losing money and negative in profit."""

    state: LimitState
    newly_triggered: bool
    daily_loss: float
    total_loss: float


def _ist_day(now: datetime) -> date:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(IST).date()


class LossLimits:
    def __init__(self, store: Store, book_cfg: BookConfig, holidays: Iterable[date]):
        self._store = store
        self._cfg = book_cfg
        self._holidays = frozenset(holidays)
        self._book = book_cfg.name
        self._baseline_key = f"baseline:{self._book}"
        self._paused_key = f"paused_on:{self._book}"
        self._hard_key = f"hard_stop:{self._book}"
        self._stop_id_key = f"stop_id:{self._book}"
        with store.transaction():
            if store.get_state(self._baseline_key) is None:
                store.set_state(self._baseline_key, float(book_cfg.capital))

    # ---- persisted values ----

    def baseline(self) -> float:
        """Total loss is measured from here. Starts at the book's capital."""
        value = self._store.get_state(self._baseline_key)
        return float(self._cfg.capital) if value is None else float(value)

    def stop_id(self) -> int:
        """Number of recovery resets so far; identifies the current hard stop's recovery plan."""
        return int(self._store.get_state(self._stop_id_key, 0))

    def _hard_stopped(self) -> bool:
        return self._store.get_state(self._hard_key) is True

    def _paused_today(self, today: date) -> bool:
        return self._store.get_state(self._paused_key) == today.isoformat()

    def _persisted_state(self, today: date) -> LimitState:
        if self._hard_stopped():
            return LimitState.HARD_STOPPED
        if self._paused_today(today):
            return LimitState.DAILY_PAUSED
        return LimitState.ACTIVE

    # ---- state machine ----

    def state(self, now: datetime) -> LimitState:
        """Current state from persisted data only. Needs no equity and writes nothing."""
        return self._persisted_state(_ist_day(now))

    def evaluate(self, equity: float, now: datetime) -> LimitEval:
        """Apply ``equity`` at ``now``, persist any transition and report the resulting state.

        ``newly_triggered`` is true only when this call moves the persisted state from a lower
        state to ``DAILY_PAUSED`` or ``HARD_STOPPED`` (a daily pause followed by a hard stop
        counts), never on a repeat evaluation and never after a restart.
        """
        if not math.isfinite(equity):
            # NaN compares false against every limit; it must not read as "no loss".
            raise ValueError(f"equity must be a finite number, got {equity!r}")
        today = _ist_day(now)

        # One transaction: the read of the latched state and its update cannot interleave with
        # another thread's evaluation, so a transition is reported to exactly one caller.
        with self._store.transaction():
            was_hard = self._hard_stopped()
            was_paused = self._paused_today(today)
            before = self._persisted_state(today)

            base = self.baseline()
            previous_close = self._store.equity_snapshot(
                self._book, previous_trading_day(today, self._holidays))
            daily_ref = base if previous_close is None else previous_close
            daily_loss = daily_ref - equity
            total_loss = base - equity

            hard = was_hard or total_loss >= self._cfg.total_loss_limit - TOLERANCE
            paused = was_paused or daily_loss >= self._cfg.daily_loss_limit - TOLERANCE

            if hard and not was_hard:
                self._store.set_state(self._hard_key, True)
                log.error("HARD STOP on %s: total loss %.2f >= limit %.2f (baseline %.2f, equity %.2f)",
                          self._book, total_loss, self._cfg.total_loss_limit, base, equity)
            if paused and not was_paused:
                self._store.set_state(self._paused_key, today.isoformat())
                log.warning("daily pause on %s %s: loss %.2f >= limit %.2f (reference %.2f, equity %.2f)",
                            self._book, today, daily_loss, self._cfg.daily_loss_limit, daily_ref, equity)

        state = (LimitState.HARD_STOPPED if hard
                 else LimitState.DAILY_PAUSED if paused
                 else LimitState.ACTIVE)
        return LimitEval(state=state, newly_triggered=state is not LimitState.ACTIVE and state != before,
                         daily_loss=daily_loss, total_loss=total_loss)

    def reset_baseline(self, equity: float) -> None:
        """Approved recovery: measure total loss from ``equity`` from now on.

        Clears the hard stop and any daily pause and bumps ``stop_id``, in one atomic write.
        """
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError(f"baseline equity must be a finite number > 0, got {equity!r}")
        with self._store.transaction():
            next_id = self.stop_id() + 1
            self._store.set_state(self._baseline_key, float(equity))
            self._store.set_state(self._hard_key, False)
            self._store.set_state(self._paused_key, None)
            self._store.set_state(self._stop_id_key, next_id)
        log.warning("baseline for %s reset to %.2f (stop id %d); hard stop and daily pause cleared",
                    self._book, equity, next_id)
