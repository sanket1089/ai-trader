"""Executor: the only path from an order to the broker, and it always goes through the RiskManager.

``submit`` flow: throttle (1 order per ``min_interval_s``), ``risk.check``, record the decision,
then one of

* rejected: the order is saved REJECTED, the owner is told the reasons, nothing reaches the broker;
* dry-run: the order is saved CANCELLED with "dry-run" appended to its reason, the owner is told,
  the broker is never called;
* otherwise: the order is saved PENDING and ``broker.try_fill`` is called; a fill is announced.

Rules every method follows:

* One ``asyncio.Lock`` serialises all broker interaction (``submit``, ``submit_exit``,
  ``reprice_exit``, ``retry_pending``), and the throttle wait happens inside it, so two callers can
  never race each other past the 1-per-second spacing. No store transaction is held across an
  ``await``.
* Orders are re-read from the store before they are retried or re-priced. A caller's copy may be
  stale (a stale PENDING copy of a filled order would make the broker raise an IntegrityError).
* Owner notifications are plain text, are sent after the lock is released, and can never fail an
  order: a down notifier is logged (type of error only, never its message) and ignored.
* In dry-run the broker is never called, for fills or for cancels.
"""

import asyncio
import logging
import math
import os
import time
import traceback
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import date, datetime
from fractions import Fraction
from typing import Protocol

from aitrader.approvals import ApprovalCard
from aitrader.broker.base import Broker
from aitrader.calendar import IST
from aitrader.models import Fill, Order, OrderStatus, Quote, Side, TradeIdea
from aitrader.risk.manager import RiskManager
from aitrader.store import Store

log = logging.getLogger(__name__)


class Notifier(Protocol):
    async def send(self, text: str) -> None: ...

    async def send_card(self, card: ApprovalCard, text: str) -> None: ...


def _rupees(x: float) -> str:
    return f"₹{x:.2f}"


def _append_reason(reason: str, extra: str) -> str:
    return f"{reason}; {extra}" if reason else extra


def _describe(o: Order) -> str:
    return f"{o.book} {o.side} {o.qty} {o.symbol} @ limit {_rupees(o.limit_price)}"


def describe_error(exc: BaseException) -> str:
    """The error's type and where it was raised, never its message (it may hold a URL or a token)."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return type(exc).__name__
    last = frames[-1]
    return f"{type(exc).__name__} at {os.path.basename(last.filename)}:{last.lineno} in {last.name}"


def marketable_limit(side: Side, price: float, band_pct: float) -> float:
    """The limit for a discretionary order (an entry, or a SELL that is not a stop/target exit) that
    can actually fill, to the paisa: the edge of the risk gate's ``band_pct`` band around ``price``.

    The paper broker fills a BUY at the price plus its slippage (a SELL at the price minus it) and only
    if that is within the limit, so a limit equal to the last price never fills; the risk gate refuses a
    limit more than ``band_pct`` from the last price. A BUY gets the top of the band (floored to the
    paisa), a SELL the bottom (ceiled), so the result is never outside the band. Mirrors the 1% through
    the price that ``exits.exit_limit`` uses for exits. Exact fraction arithmetic keeps float noise
    from moving it a paisa.
    """
    exact = Fraction(price) * (1 + Fraction(str(band_pct)) / 100 * (1 if side == Side.BUY else -1)) * 100
    return (math.floor(exact) if side == Side.BUY else math.ceil(exact)) / 100


def _ist_day(now: datetime) -> date:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(IST).date()


class Executor:
    def __init__(self, store: Store, broker: Broker, risk: RiskManager, notifier: Notifier,
                 dry_run: bool, min_interval_s: float = 1.0,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self._store = store
        self._broker = broker
        self._risk = risk
        self._notifier = notifier
        self._dry_run = dry_run
        self._min_interval = min_interval_s
        self._sleep = sleep
        self._clock = clock
        self._lock = asyncio.Lock()
        self._last_sent: float | None = None      # clock reading of the last order that went out

    # ---- public API ----

    async def submit(self, order: Order, quote: Quote | None, now: datetime,
                     idea: TradeIdea | None = None) -> Fill | None:
        """Run a new order through risk and, if approved, to the broker. Returns the fill, if any."""
        msgs: list[str] = []
        try:
            async with self._lock:
                return await self._attempt(order, quote, now, idea, msgs, is_new=True)
        finally:
            await self._flush(msgs)

    async def submit_exit(self, order: Order, quote: Quote | None, now: datetime) -> Fill | None:
        """Place a stop/target exit (an ``is_exit`` SELL).

        First cancels this book's PENDING discretionary SELLs of the same symbol, because they
        reserve the holding and the risk gate would otherwise refuse the exit. If an exit for the
        same holding is already PENDING, nothing new is created and ``None`` is returned: re-pricing
        the existing one is ``reprice_exit``'s job.
        """
        if order.side is not Side.SELL or not order.is_exit:
            raise ValueError("submit_exit takes an exit SELL")
        msgs: list[str] = []
        try:
            async with self._lock:
                if not self._dry_run:
                    pending = self._store.pending_orders(order.book)
                    if any(p.is_exit and p.side is Side.SELL and p.symbol == order.symbol for p in pending):
                        log.info("%s %s already has a pending exit; not creating another",
                                 order.book, order.symbol)
                        return None
                    for p in pending:
                        if p.side is Side.SELL and not p.is_exit and p.symbol == order.symbol:
                            self._cancel(p, "cancelled: superseded by a stop/target exit")
                            msgs.append(f"Cancelled {_describe(p)}: superseded by a stop/target exit")
                return await self._attempt(order, quote, now, None, msgs, is_new=True)
        finally:
            await self._flush(msgs)

    async def reprice_exit(self, order_id: str, limit_price: float, quote: Quote | None,
                           now: datetime) -> Fill | None:
        """Re-price the PENDING exit ``order_id`` (same order: new limit, ``attempts`` + 1) and try
        the fill again, through the risk gate. A missing or no-longer-pending order is left alone."""
        if self._dry_run:
            return None
        msgs: list[str] = []
        try:
            async with self._lock:
                order = self._store.get_order(order_id)             # never a caller's copy
                if order is None or order.status is not OrderStatus.PENDING or not order.is_exit:
                    log.info("order %s is not a pending exit; not re-pricing it", order_id)
                    return None
                order.limit_price = limit_price
                order.attempts += 1
                self._store.update_order(order)
                return await self._attempt(order, quote, now, None, msgs, is_new=False)
        finally:
            await self._flush(msgs)

    async def retry_pending(self, quotes: Mapping[str, Quote], now: datetime) -> list[Fill]:
        """Cancel PENDING non-exit orders left from an earlier IST day; retry today's through the
        risk gate and the broker. Exits are the ExitWatcher's, not touched here.

        A retried order the risk gate now refuses stays PENDING (its limit may become valid again,
        and a message on every retry would be noise); the decision is recorded.

        An order whose retry raises is logged (error type and place, book, symbol; never the
        message), the owner is told once per IST day per book and symbol, and the pass carries on.
        """
        if self._dry_run:
            return []
        today = _ist_day(now)
        fills: list[Fill] = []
        msgs: list[str] = []
        failed: list[tuple[Order, str]] = []                        # (order, error type) to alert about
        try:
            async with self._lock:
                for listed in self._store.pending_orders():
                    try:
                        fill = await self._retry_one(listed.id, quotes, now, today, msgs)
                    except Exception as exc:  # noqa: BLE001 - one bad order must not stop the pass
                        log.error("retrying order %s (%s %s) failed: %s", listed.id, listed.book,
                                  listed.symbol, describe_error(exc))
                        failed.append((listed, type(exc).__name__))
                        continue
                    if fill is not None:
                        fills.append(fill)
        finally:
            await self._flush(msgs)
            for order, error in failed:
                await self.notify_once(
                    f"check_error_alerted:{order.book}:{order.symbol}:{today.isoformat()}",
                    f"{order.book} {order.symbol}: error ({error}) while retrying pending order "
                    f"{order.id}; it stays pending and the other orders carried on")
        return fills

    def cancel_order(self, order_id: str, why: str) -> bool:
        """Cancel one PENDING order (the owner's /stop): tell the broker, then record CANCELLED with
        ``why`` appended to the reason. Returns False if the order is missing or no longer pending.

        Synchronous on purpose: with no ``await`` between the read and the write, nothing else on the
        event loop can fill the order in between, so the kill switch cannot race a fill. It takes no
        lock for the same reason (the lock only serialises the throttle waits). The owner is not
        messaged here; the caller reports.
        """
        order = self._store.get_order(order_id)             # never a caller's copy
        if order is None or order.status is not OrderStatus.PENDING:
            return False
        self._cancel(order, why)
        return True

    async def notify(self, text: str) -> bool:
        """Send ``text`` to the owner. Never raises (a down notifier must not break trading);
        returns whether the notifier accepted it."""
        try:
            await self._notifier.send(text)
        except Exception as exc:  # noqa: BLE001 - only the type is logged: the message may hold a URL/token
            log.warning("notification failed: %s", type(exc).__name__)
            return False
        return True

    async def notify_once(self, key: str, text: str) -> bool:
        """Send ``text`` unless the state ``key`` is already set, and set it only once the message
        went through, so an alert that could not be delivered is tried again. Returns True if sent."""
        if self._store.get_state(key):
            return False
        if not await self.notify(text):
            return False
        self._store.set_state(key, True)
        return True

    # ---- internals (call with the lock held) ----

    async def _retry_one(self, order_id: str, quotes: Mapping[str, Quote], now: datetime, today: date,
                         msgs: list[str]) -> Fill | None:
        """Cancel or retry one listed PENDING order, working from the stored copy of it."""
        order = self._store.get_order(order_id)                     # fresh: the lock may have waited
        if order is None or order.status is not OrderStatus.PENDING or order.is_exit:
            return None
        created = order.created_at.astimezone(IST).date()
        if created < today:
            self._cancel(order, f"cancelled: left over from {created.isoformat()}")
            msgs.append(f"Cancelled {_describe(order)}: left over from {created.isoformat()}")
            return None
        quote = quotes.get(order.symbol)
        if quote is None:
            log.warning("no quote for %s; not retrying order %s", order.symbol, order.id)
            return None
        idea = self._store.get_idea(order.idea_id) if order.idea_id is not None else None
        return await self._attempt(order, quote, now, idea, msgs, is_new=False)

    async def _throttle(self) -> None:
        """Wait until ``min_interval_s`` has passed since the last order went out."""
        if self._last_sent is not None:
            wait = self._min_interval - (self._clock() - self._last_sent)
            if wait > 0:
                await self._sleep(wait)
        self._last_sent = self._clock()

    async def _attempt(self, order: Order, quote: Quote | None, now: datetime, idea: TradeIdea | None,
                       msgs: list[str], *, is_new: bool) -> Fill | None:
        """Throttle, risk-check and (if approved) try to fill ``order``.

        ``is_new``: the order is not in the store yet (it is saved here, as REJECTED, CANCELLED for
        dry-run, or PENDING). Otherwise it is a stored PENDING order being retried.
        """
        await self._throttle()
        decision = self._risk.check(order, quote, now, idea)
        self._store.save_risk_decision(order.id, decision.approved, decision.reasons, now)

        if not decision.approved:
            reasons = "; ".join(decision.reasons)
            if is_new:
                order.status = OrderStatus.REJECTED
                self._store.save_order(order)
                msgs.append(f"Rejected by risk: {_describe(order)}: {reasons}")
            else:
                log.info("order %s stays pending, risk refused the retry: %s", order.id, reasons)
            return None

        if is_new:
            if self._dry_run:
                order.status = OrderStatus.CANCELLED
                order.reason = _append_reason(order.reason, "dry-run")
                self._store.save_order(order)
                msgs.append(f"Dry-run: not placing {_describe(order)} ({order.reason})")
                return None
            self._store.save_order(order)

        fill = self._broker.try_fill(order, quote)
        if fill is not None:
            text = (f"Filled {fill.book} {fill.side} {fill.qty} {fill.symbol} @ {_rupees(fill.price)}, "
                    f"charges {_rupees(fill.charges)}")
            msgs.append(f"{text} ({order.reason})" if fill.is_exit and order.reason else text)
            return fill
        stored = self._store.get_order(order.id)
        if stored is not None and stored.status is OrderStatus.REJECTED:
            msgs.append(f"Order rejected by the broker: {_describe(order)} (see the log)")
        return None

    def _cancel(self, order: Order, why: str) -> None:
        """Cancel a PENDING order: tell the broker, then record CANCELLED with the reason appended."""
        self._broker.cancel(order.id)
        self._store.update_order(replace(order, status=OrderStatus.CANCELLED,
                                         reason=_append_reason(order.reason, why)))

    async def _flush(self, msgs: list[str]) -> None:
        for text in msgs:
            await self.notify(text)
