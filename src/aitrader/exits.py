"""ExitWatcher: sells automatically when a position reaches its stop or its target.

``check(now)`` runs every ``exit_check_minutes`` and does nothing outside ``exit_start``..15:30 IST,
so a gap-down at the open is caught by the first check at or after 09:15. For each watched position
with a usable quote:

* a PENDING exit for it exists: once ``exit_reprice_minutes`` have passed since it was last priced
  (state key ``exit_repriced_at:<order_id>``, the order's ``created_at`` for the first time) it is
  re-priced at the current price (the same order, ``attempts`` + 1) and the fill is tried again.
  There is no cap, so an exit never sits at a stale limit, also across days. When ``attempts`` first
  exceeds ``exit_max_reprices`` the owner is told once (state key ``exit_alerted:<order_id>``) and
  re-pricing carries on;
* otherwise, if ``price <= stop`` or ``price >= target``, an exit SELL of the whole position is
  placed through the Executor (and so through the RiskManager, which never lets loss limits, the
  sell-day cap or the AI budget block an exit).

A stop or target of 0.0 means *unset*: it never triggers (a target of 0 would "always" be hit).
A position with no stop gets one owner alert (state key ``nostop_alerted:<book>:<symbol>``).

A position with no quote, or only a stale or unusable one, is skipped with a warning: the risk gate
refuses a stale quote anyway, so trying would only fill the log and the owner's chat with
rejections. Its stop-loss cannot be checked, so the owner is told once per IST day (state key
``noquote_alerted:<book>:<symbol>:<date>``).

One position that raises (a broker or risk error) is logged by error type, place, book and symbol
(never the message), the owner is told once per IST day (``check_error_alerted:<book>:<symbol>:<date>``)
and the pass carries on with the other positions.

Every once-only alert is marked as sent only after the notifier accepted it.
"""

import asyncio
import logging
import math
import uuid
from datetime import datetime, timedelta

from aitrader.calendar import IST, in_window
from aitrader.config import RiskConfig
from aitrader.data.market_data import MarketData, is_stale
from aitrader.execution import Executor, describe_error
from aitrader.models import Fill, Order, OrderStatus, Position, Quote, Side
from aitrader.risk.manager import MARKET_CLOSE
from aitrader.store import Store

log = logging.getLogger(__name__)

# The risk gate accepts a limit down to ``price x (1 - band) - 1e-6``; stay inside that.
_BAND_SLACK = 1e-7


def exit_limit(price: float, band_pct: float) -> float:
    """Marketable limit for an exit: ``round(price x (1 - band), 2)``.

    Rounding to the nearest paisa can land up to half a paisa *below* the band floor the risk gate
    enforces (250.51 x 0.99 = 248.0049 rounds to 248.00), and the gate would refuse the exit. In
    that case the limit steps up one paisa, to the first price at or above the floor.
    """
    floor = price * (1 - band_pct / 100)
    limit = round(floor, 2)
    if limit < floor - _BAND_SLACK:
        limit = round(limit + 0.01, 2)
    return limit


def _day(now: datetime) -> str:
    """The IST calendar date of ``now`` as an ISO string (the day part of the once-a-day alert keys)."""
    return now.astimezone(IST).date().isoformat()


class ExitWatcher:
    def __init__(self, store: Store, executor: Executor, market_data: MarketData, risk_cfg: RiskConfig,
                 books: list[str]):
        self._store = store
        self._executor = executor
        self._market_data = market_data
        self._cfg = risk_cfg
        self._books = list(books)
        self._lock = asyncio.Lock()        # two overlapping checks must not both place an exit

    async def check(self, now: datetime) -> list[Fill]:
        """One pass over every position of the watched books. Returns the exit fills it produced."""
        if not in_window(now, self._cfg.exit_start, MARKET_CLOSE):
            return []
        async with self._lock:
            held = [(book, p.symbol) for book in self._books for p in self._store.positions(book)]
            if not held:
                return []
            symbols = sorted({symbol for _, symbol in held})
            # the market-data layer is blocking (network), so keep it off the event loop
            quotes = await asyncio.to_thread(self._market_data.quotes, symbols)
            fills: list[Fill] = []
            for book, symbol in held:
                try:
                    fill = await self._check_position(book, symbol, quotes.get(symbol), now)
                except Exception as exc:  # noqa: BLE001 - one bad position must not stop the pass
                    log.error("exit check for %s %s failed: %s", book, symbol, describe_error(exc))
                    await self._executor.notify_once(
                        f"check_error_alerted:{book}:{symbol}:{_day(now)}",
                        f"{book} {symbol}: error ({type(exc).__name__}) while checking its stop and "
                        f"target; the other positions carried on")
                    continue
                if fill is not None:
                    fills.append(fill)
            return fills

    # ---- one position ----

    async def _check_position(self, book: str, symbol: str, quote: Quote | None,
                              now: datetime) -> Fill | None:
        position = next((p for p in self._store.positions(book) if p.symbol == symbol), None)
        if position is None:                      # sold since the pass began
            return None
        await self._alert_if_no_stop(position)

        if quote is None:
            log.warning("no quote for %s %s; skipping its exit check", book, symbol)
        elif not (math.isfinite(quote.price) and quote.price > 0):
            log.warning("unusable price %r for %s %s; skipping its exit check", quote.price, book, symbol)
        elif is_stale(quote, now, self._cfg):
            log.warning("quote for %s %s is stale (%s); skipping its exit check", book, symbol, quote.ts)
        else:
            return await self._act_on_price(position, quote, now)
        await self._executor.notify_once(f"noquote_alerted:{book}:{symbol}:{_day(now)}",
                                         f"no fresh price for {symbol}; its stop-loss can't be checked")
        return None

    async def _act_on_price(self, position: Position, quote: Quote, now: datetime) -> Fill | None:
        book, symbol = position.book, position.symbol

        pending = self._pending_exit(book, symbol)
        if pending is not None:
            return await self._handle_pending(pending, quote, now)

        reason = self._trigger(position, quote.price)
        if reason is None:
            return None
        order = Order(id=uuid.uuid4().hex, book=book, symbol=symbol, side=Side.SELL, qty=position.qty,
                      limit_price=exit_limit(quote.price, self._cfg.exit_band_pct), reason=reason,
                      is_exit=True, idea_id=position.idea_id, created_at=now,
                      status=OrderStatus.PENDING, attempts=0)
        return await self._executor.submit_exit(order, quote, now)

    @staticmethod
    def _trigger(position: Position, price: float) -> str | None:
        """Why this position should be sold at ``price``, or ``None``. A level of 0 is unset."""
        if position.stop > 0 and price <= position.stop:
            return f"stop hit at ₹{price:.2f}"
        if position.target > 0 and price >= position.target:
            return f"target hit at ₹{price:.2f}"
        return None

    def _pending_exit(self, book: str, symbol: str) -> Order | None:
        """The PENDING exit SELL for this holding, read fresh from the store."""
        return next((o for o in self._store.pending_orders(book)
                     if o.is_exit and o.side is Side.SELL and o.symbol == symbol), None)

    def _last_priced(self, order: Order) -> datetime:
        """When the order was last priced: its last re-price, or its creation if never re-priced."""
        raw = self._store.get_state(f"exit_repriced_at:{order.id}")
        if isinstance(raw, str):
            try:
                return max(order.created_at, datetime.fromisoformat(raw))
            except ValueError:
                log.warning("ignoring a bad re-price time %r for order %s", raw, order.id)
        return order.created_at

    async def _handle_pending(self, order: Order, quote: Quote, now: datetime) -> Fill | None:
        """Re-price the pending exit at the current price if it was last priced long enough ago."""
        if now - self._last_priced(order) < timedelta(minutes=self._cfg.exit_reprice_minutes):
            return None                                           # give it time to fill first
        limit = exit_limit(quote.price, self._cfg.exit_band_pct)
        fill = await self._executor.reprice_exit(order.id, limit, quote, now)
        self._store.set_state(f"exit_repriced_at:{order.id}", now.astimezone(IST).isoformat())
        if fill is None:
            stored = self._store.get_order(order.id)
            if (stored is not None and stored.status is OrderStatus.PENDING
                    and stored.attempts > self._cfg.exit_max_reprices):
                await self._alert_unfilled(stored)
        return fill

    # ---- owner alerts (each at most once) ----

    async def _alert_unfilled(self, order: Order) -> None:
        await self._executor.notify_once(
            f"exit_alerted:{order.id}",
            f"{order.book} {order.symbol}: exit order {order.id} ({order.reason}) is still not filled "
            f"after {order.attempts} re-prices; it is re-priced at every check and stays pending "
            f"(limit ₹{order.limit_price:.2f}). Please check it.")

    async def _alert_if_no_stop(self, position: Position) -> None:
        key = f"nostop_alerted:{position.book}:{position.symbol}"
        if position.stop > 0:
            if self._store.get_state(key):
                self._store.set_state(key, False)                 # re-arm for if it is ever lost again
            return
        await self._executor.notify_once(key, f"{position.book} {position.symbol} has no stop-loss set")
