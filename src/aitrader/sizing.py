"""Position sizing: how many whole units of an idea each book takes.

Code, not the model, decides quantity. All money math rounds down, so a size never costs more
than the budget, and the result is always an ``int >= 0`` (0 means "skip this idea").
"""

import math
from fractions import Fraction

from aitrader.charges import compute_charges
from aitrader.config import BookConfig, ChargesConfig
from aitrader.models import Side, TradeIdea


def _usable(x: float) -> bool:
    return math.isfinite(x) and x > 0


def size_idea(idea: TradeIdea, book: BookConfig, cash: float, equity: float, held_qty: int,
              charges_cfg: ChargesConfig) -> int:
    """Units to order for ``idea`` in ``book``.

    BUY: floor(budget / limit_price) with budget = min(cash, max_position_pct% of equity), then
    reduced until the value plus the BUY charges fits in the budget, because the broker needs
    cash for both. 0 if not even one unit fits.
    SELL: all of ``held_qty`` (never negative). An idea whose ``limit_price`` is not a finite
    number above 0 gets 0 on either side, and so does a BUY with unusable cash, equity or cap, or
    a side other than BUY/SELL.
    """
    if not _usable(idea.limit_price):
        return 0
    if idea.side == Side.SELL:
        return max(0, int(held_qty))
    if idea.side != Side.BUY:
        return 0
    if not (_usable(cash) and _usable(equity) and _usable(book.max_position_pct)):
        return 0
    budget = min(cash, book.max_position_pct * equity / 100)
    # Exact rational floor: float division can round up to a whole unit that costs a hair more
    # than the budget (1.1 / 0.1 is 11.000000000000002).
    qty = max(0, math.floor(Fraction(budget) / Fraction(idea.limit_price)))
    while qty > 0 and (qty * idea.limit_price
                       + compute_charges(Side.BUY, qty, idea.limit_price, charges_cfg).total) > budget:
        qty -= 1
    return qty
