"""Paper broker: simulated NSE delivery (CNC) fills against delayed quotes.

Rules:

* A BUY fills at ``quote.price x (1 + slippage%)`` if that is at or below the limit; a SELL at
  ``quote.price x (1 - slippage%)`` if that is at or above the limit. The fill price is rounded to
  the paisa, and the limit is compared against the rounded price actually paid.
* Every order produces at most one ``Fill`` for its whole quantity (no partial fills).
* Real delivery charges apply (``compute_charges``), including the DP charge on sells.
* Cash and position quantity can never go negative: an order that would do that is REJECTED.
* The fill's timestamp is ``quote.ts``. The broker has no clock of its own; the risk gate has
  already refused stale quotes, so the price and its time stay together.
* The order is saved first (PENDING) when the caller has not already saved it. All the writes of
  a fill (fill row, order status, position, cash) then commit together or not at all.
"""

import logging
import math
from dataclasses import replace

from aitrader.charges import compute_charges
from aitrader.config import ChargesConfig
from aitrader.models import Fill, Order, OrderStatus, Position, Quote, Side
from aitrader.store import Store

log = logging.getLogger(__name__)


class PaperBroker:
    def __init__(self, store: Store, charges_cfg: ChargesConfig, slippage_pct: float = 0.30):
        # 0.30 = a base of 0.05 plus the extra 0.25 required by the delayed feed (spec 7.3).
        if not 0 <= slippage_pct < 100:
            raise ValueError(f"slippage_pct must be in [0, 100), got {slippage_pct}")
        self.store = store
        self.charges_cfg = charges_cfg
        self.slippage_pct = slippage_pct

    def try_fill(self, order: Order, quote: Quote) -> Fill | None:
        if order.status is not OrderStatus.PENDING:
            log.info("order %s is %s, not PENDING; nothing to fill", order.id, order.status)
            return None
        self._ensure_saved(order)

        if order.qty < 1:
            return self._reject(order, f"qty must be >= 1, got {order.qty}")
        buy = order.side is Side.BUY
        held = next((p for p in self.store.positions(order.book) if p.symbol == order.symbol), None)
        held_qty = held.qty if held else 0
        if not buy and order.qty > held_qty:
            return self._reject(order, f"sell qty {order.qty} exceeds the {held_qty} held")

        if quote.symbol != order.symbol:
            log.warning("order %s (%s) ignored a quote for %s", order.id, order.symbol, quote.symbol)
            return None
        slip = self.slippage_pct / 100
        price = round(quote.price * (1 + slip if buy else 1 - slip), 2)
        if not 0 < price < math.inf:                       # also catches NaN
            log.warning("order %s: unusable quote price %r for %s", order.id, quote.price, order.symbol)
            return None
        if (price > order.limit_price) if buy else (price < order.limit_price):
            return None                                    # limit not met: stays PENDING

        value = order.qty * price
        charges = compute_charges(order.side, order.qty, price, self.charges_cfg).total
        cash = self.store.cash(order.book)
        new_cash = cash - (value + charges) if buy else cash + (value - charges)
        if new_cash < 0:
            return self._reject(
                order, f"cash {cash:.2f} cannot cover {order.side} of {value:.2f} with charges {charges:.2f}")

        position = self._buy_position(order, quote, price, held) if buy else self._sell_position(order, held)
        fill = Fill(order_id=order.id, book=order.book, symbol=order.symbol, side=order.side,
                    qty=order.qty, price=price, charges=charges, is_exit=order.is_exit, ts=quote.ts)

        with self.store.transaction():
            self.store.save_fill(fill)
            self.store.update_order(replace(order, status=OrderStatus.FILLED))
            if position is None:
                self.store.remove_position(order.book, order.symbol)
            else:
                self.store.upsert_position(position)
            self.store.set_cash(order.book, new_cash)
        order.status = OrderStatus.FILLED                  # only once the commit has succeeded
        log.info("filled %s %s %d %s @ %.2f (charges %.2f)", order.id, order.side, order.qty,
                 order.symbol, price, charges)
        return fill

    def cancel(self, order_id: str) -> None:
        for o in self.store.pending_orders():
            if o.id == order_id:
                o.status = OrderStatus.CANCELLED
                self.store.update_order(o)
                return
        log.info("cancel: order %s is not pending; nothing to do", order_id)

    # ---- helpers ----

    def _ensure_saved(self, order: Order) -> None:
        """Save a new order as PENDING. One the caller (the Executor) already saved is left alone.

        An order whose object says PENDING but whose stored row is not (a stale copy) is not in
        ``pending_orders``, so the insert fails on the primary key rather than silently re-filling it.
        """
        if not any(o.id == order.id for o in self.store.pending_orders(order.book)):
            self.store.save_order(order)

    def _reject(self, order: Order, reason: str) -> None:
        log.warning("order %s rejected: %s", order.id, reason)
        self.store.update_order(replace(order, status=OrderStatus.REJECTED))
        order.status = OrderStatus.REJECTED
        return None

    def _buy_position(self, order: Order, quote: Quote, price: float, held: Position | None) -> Position:
        """The position after a BUY: new, or the existing one averaged up.

        Stop and target come from the order's idea. If there is none, a new position gets 0.0 for
        both, and an existing one keeps its levels (so protection is never weakened by a lookup miss).
        """
        idea = self.store.get_idea(order.idea_id) if order.idea_id is not None else None
        if idea is None:
            log.warning("order %s: no idea %r found, so the stop and target cannot be set",
                        order.id, order.idea_id)
        if held is None:
            return Position(book=order.book, symbol=order.symbol, qty=order.qty, avg_price=price,
                            stop=idea.stop if idea else 0.0, target=idea.target if idea else 0.0,
                            opened_at=quote.ts, idea_id=order.idea_id)
        qty = held.qty + order.qty
        avg_price = (held.qty * held.avg_price + order.qty * price) / qty
        if idea is None:
            return replace(held, qty=qty, avg_price=avg_price)
        return replace(held, qty=qty, avg_price=avg_price, stop=idea.stop, target=idea.target,
                       idea_id=order.idea_id)

    @staticmethod
    def _sell_position(order: Order, held: Position | None) -> Position | None:
        """The position after a SELL, or ``None`` when it is closed out (the caller checked qty <= held)."""
        remaining = held.qty - order.qty
        return replace(held, qty=remaining) if remaining > 0 else None
