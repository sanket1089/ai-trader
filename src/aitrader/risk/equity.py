"""Mark-to-market equity of a book.

Equity = cash + sum(qty x price) - sum(estimated sell charges per holding). The sell charges
include the flat DP charge, so equity is what the book would hold if every position were
liquidated now. That is deliberately pessimistic: a position can never look more valuable than
it is, which is the safe direction for the loss limits.
"""

import logging
import math

from aitrader.charges import compute_charges
from aitrader.config import ChargesConfig
from aitrader.models import Quote, Side
from aitrader.store import Store

log = logging.getLogger(__name__)


def _usable(price: float) -> bool:
    return math.isfinite(price) and price > 0


def book_equity(store: Store, book: str, quotes: dict[str, Quote], charges_cfg: ChargesConfig) -> float:
    """Equity of ``book`` marked at ``quotes``.

    A holding with no usable quote (missing, NaN, infinite or <= 0) is marked at its
    ``avg_price`` and a warning is logged, so equity stays computable but the gap is visible.
    An unseeded book raises ``KeyError`` (from ``Store.cash``) rather than reading as zero cash.
    """
    # Cash and positions are read together: a fill commits both in one transaction, and the
    # store lock held here keeps another thread from landing one between the two reads.
    with store.transaction():
        equity = store.cash(book)
        positions = store.positions(book)

    for p in positions:
        quote = quotes.get(p.symbol)
        if quote is not None and _usable(quote.price):
            price = quote.price
        else:
            price = p.avg_price
            log.warning("no usable quote for %s in book %s; marking it at its average price %.2f",
                        p.symbol, book, price)
        equity += p.qty * price - compute_charges(Side.SELL, p.qty, price, charges_cfg).total
    return equity
