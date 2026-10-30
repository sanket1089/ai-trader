"""ExitWatcher: sells automatically when a position reaches its stop or its target.

``check(now)`` runs every ``exit_check_minutes`` and does nothing outside ``exit_start``..15:30 IST,
so a gap-down at the open is caught by the first check at or after 09:15. For each watched position
with a usable quote:

* a PENDING exit for it exists: once it is ``exit_reprice_minutes`` old it is re-priced at the
  current price (the same order, ``attempts`` + 1); after ``exit_max_reprices`` re-prices it is
  left pending and the owner is told once (state key ``exit_alerted:<order_id>``);
* otherwise, if ``price <= stop`` or ``price >= target``, an exit SELL of the whole position is
  placed through the Executor (and so through the RiskManager, which never lets loss limits, the
  sell-day cap or the AI budget block an exit).

A stop or target of 0.0 means *unset*: it never triggers (a target of 0 would "always" be hit).
A position with no stop gets one owner alert (state key ``nostop_alerted:<book>:<symbol>``).

A position with no quote, or only a stale one, is skipped with a warning: the risk gate refuses a
stale quote anyway, so trying would only fill the log and the owner's chat with rejections.
"""

import asyncio
import logging
import math
import uuid
from datetime import datetime, timedelta

from aitrader.calendar import in_window
from aitrader.config import RiskConfig
from aitrader.data.market_data import MarketData, is_stale
from aitrader.execution import Executor
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
                fill = await self._check_position(book, symbol, quotes.get(symbol), now)
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
            return None
        if not (math.isfinite(quote.price) and quote.price > 0):
            log.warning("unusable price %r for %s %s; skipping its exit check", quote.price, book, symbol)
            return None
        if is_stale(quote, now, self._cfg):
            log.warning("quote for %s %s is stale (%s); skipping its exit check", book, symbol, quote.ts)
            return None

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

    async def _handle_pending(self, order: Order, quote: Quote, now: datetime) -> Fill | None:
        if now - order.created_at < timedelta(minutes=self._cfg.exit_reprice_minutes):
            return None                                           # give it time to fill first
        if order.attempts >= self._cfg.exit_max_reprices:
            await self._alert_unfilled(order)
            return None
        limit = exit_limit(quote.price, self._cfg.exit_band_pct)
        return await self._executor.reprice_exit(order.id, limit, quote, now)

    # ---- owner alerts (each at most once) ----

    async def _alert_unfilled(self, order: Order) -> None:
        key = f"exit_alerted:{order.id}"
        if self._store.get_state(key):
            return
        self._store.set_state(key, True)
        await self._executor.notify(
            f"{order.book} {order.symbol}: exit order {order.id} ({order.reason}) is still not filled "
            f"after {order.attempts} re-prices; it stays pending at limit ₹{order.limit_price:.2f}. "
            f"Please check it.")

    async def _alert_if_no_stop(self, position: Position) -> None:
        key = f"nostop_alerted:{position.book}:{position.symbol}"
        if position.stop > 0:
            if self._store.get_state(key):
                self._store.set_state(key, False)                 # re-arm for if it is ever lost again
            return
        if self._store.get_state(key):
            return
        self._store.set_state(key, True)
        await self._executor.notify(f"{position.book} {position.symbol} has no stop-loss set")
