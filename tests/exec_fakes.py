"""Fakes shared by the executor and exit-watcher tests (importable because pytest puts ``tests/``
on ``sys.path``)."""

import asyncio
from dataclasses import replace
from datetime import datetime

from aitrader.models import Fill, Order, OrderStatus, Quote
from aitrader.risk.manager import RiskDecision
from aitrader.store import Store


class FakeTime:
    """A clock that only moves when a fake sleep moves it, so nothing really waits.

    ``sleep`` models real time: it yields to the event loop, then the clock is at least
    ``start + duration`` (two tasks sleeping together wake at the same moment, as they would in life).
    """

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        target = self.now + duration
        await asyncio.sleep(0)
        self.now = max(self.now, target)


class FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.cards: list[tuple[object, str]] = []
        self.fail = False

    async def send(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("telegram is down")
        self.texts.append(text)

    async def send_card(self, card, text: str) -> None:
        self.cards.append((card, text))


class FakeBroker:
    """Records every call. ``fills`` makes try_fill fill at the quote; ``rejects`` makes it reject."""

    def __init__(self, store: Store | None = None, fills: bool = False, rejects: bool = False,
                 raise_for: set[str] | None = None):
        self.store = store
        self.fills = fills
        self.rejects = rejects
        self.raise_for = raise_for or set()      # symbols whose try_fill blows up
        self.calls: list[tuple[Order, Quote]] = []     # order snapshots as they were when called
        self.cancelled: list[str] = []

    def try_fill(self, order: Order, quote: Quote) -> Fill | None:
        self.calls.append((replace(order), quote))
        if order.symbol in self.raise_for:
            raise RuntimeError(f"broker exploded for {order.symbol}: secret-token-123")
        if self.rejects:
            order.status = OrderStatus.REJECTED
            if self.store is not None:
                self.store.update_order(replace(order))
            return None
        if not self.fills:
            return None
        fill = Fill(order_id=order.id, book=order.book, symbol=order.symbol, side=order.side,
                    qty=order.qty, price=quote.price, charges=1.5, is_exit=order.is_exit, ts=quote.ts)
        order.status = OrderStatus.FILLED
        if self.store is not None:
            self.store.update_order(replace(order))
        return fill

    def cancel(self, order_id: str) -> None:
        self.cancelled.append(order_id)


class FakeRisk:
    """Approves (or rejects with ``reasons``) every order and records what it was asked."""

    def __init__(self, approve: bool = True, reasons: list[str] | None = None,
                 raise_for: set[str] | None = None):
        self.approve = approve
        self.reasons = reasons or []
        self.raise_for = raise_for or set()      # symbols whose check blows up
        self.calls: list[tuple[Order, Quote | None, datetime, object]] = []

    def check(self, order: Order, quote: Quote | None, now: datetime, idea=None) -> RiskDecision:
        self.calls.append((replace(order), quote, now, idea))
        if order.symbol in self.raise_for:
            raise RuntimeError(f"risk exploded for {order.symbol}: secret-token-123")
        return RiskDecision(self.approve, list(self.reasons))


class FakeMarketData:
    def __init__(self) -> None:
        self.by_symbol: dict[str, Quote] = {}
        self.calls: list[list[str]] = []

    def set(self, symbol: str, price: float, ts: datetime) -> None:
        self.by_symbol[symbol] = Quote(symbol=symbol, price=price, ts=ts)

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        self.calls.append(list(symbols))
        return {s: q for s, q in self.by_symbol.items() if s in symbols}
