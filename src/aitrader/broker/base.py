"""The one interface every trade goes through. Phase 1 has a single implementation: ``PaperBroker``."""

from typing import Protocol

from aitrader.models import Fill, Order, Quote


class Broker(Protocol):
    def try_fill(self, order: Order, quote: Quote) -> Fill | None:
        """Try to execute a PENDING order against ``quote``.

        Returns the single ``Fill`` (orders are never part-filled), or ``None`` when the order
        did not fill: it stays PENDING if the limit was not met, or ends REJECTED if it was invalid.
        """
        ...

    def cancel(self, order_id: str) -> None:
        """Cancel a PENDING order. Unknown or already finished orders are left as they are."""
        ...
