import math

import pytest

from aitrader.config import BookConfig
from aitrader.models import Side, TradeIdea
from aitrader.sizing import size_idea

REALISTIC = BookConfig(name="realistic", capital=1000.0, daily_loss_limit=100.0, total_loss_limit=500.0,
                       max_entries_per_day=1, max_sell_days_per_week=3, max_position_pct=100.0)
LEARNING = BookConfig(name="learning", capital=100000.0, daily_loss_limit=10000.0,
                      total_loss_limit=50000.0, max_entries_per_day=3, max_sell_days_per_week=None,
                      max_position_pct=20.0)


def idea(side=Side.BUY, limit_price=280.0):
    return TradeIdea(id="i1", symbol="NIFTYBEES", side=side, limit_price=limit_price, stop=270.0,
                     target=300.0, expected_move_pct=3.5, confidence=0.7, holding_days=5, reason="trend")


def test_size_realistic_takes_all_cash():
    assert size_idea(idea(), REALISTIC, cash=1000.0, equity=1000.0, held_qty=0) == 3


def test_size_learning_caps_20pct():
    assert size_idea(idea(), LEARNING, cash=100000.0, equity=100000.0, held_qty=0) == 71  # floor(20000/280)


def test_size_learning_is_limited_by_cash_when_cash_is_below_the_cap():
    assert size_idea(idea(), LEARNING, cash=5000.0, equity=100000.0, held_qty=0) == 17  # floor(5000/280)


def test_size_zero_when_unaffordable():
    assert size_idea(idea(limit_price=1500.0), REALISTIC, cash=1000.0, equity=1000.0, held_qty=0) == 0
    assert size_idea(idea(), LEARNING, cash=100.0, equity=100000.0, held_qty=0) == 0


def test_size_exact_affordability_boundary():
    assert size_idea(idea(limit_price=250.0), REALISTIC, cash=1000.0, equity=1000.0, held_qty=0) == 4
    assert size_idea(idea(limit_price=250.01), REALISTIC, cash=1000.0, equity=1000.0, held_qty=0) == 3


def test_size_never_spends_more_than_the_budget_despite_float_error():
    # 1.1 / 0.1 is 11.000000000000002 in floating point, but 11 units cost 1.1000000000000001 > 1.1.
    qty = size_idea(idea(limit_price=0.1), REALISTIC, cash=1.1, equity=1.1, held_qty=0)
    assert qty * 0.1 <= 1.1


def test_size_sell_returns_held_qty():
    assert size_idea(idea(Side.SELL), LEARNING, cash=0.0, equity=100000.0, held_qty=7) == 7
    assert size_idea(idea(Side.SELL), REALISTIC, cash=0.0, equity=1000.0, held_qty=0) == 0


def test_size_sell_never_negative():
    assert size_idea(idea(Side.SELL), LEARNING, cash=0.0, equity=100000.0, held_qty=-3) == 0


@pytest.mark.parametrize("price", [0.0, -5.0, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_size_invalid_limit_price_is_zero(price, side):
    assert size_idea(idea(side, price), REALISTIC, cash=1000.0, equity=1000.0, held_qty=5) == 0


@pytest.mark.parametrize("cash,equity", [(-100.0, 1000.0), (math.nan, 1000.0), (1000.0, math.nan),
                                         (math.inf, 1000.0), (1000.0, math.inf), (1000.0, -1000.0),
                                         (0.0, 1000.0)])
def test_size_buy_with_bad_money_is_zero(cash, equity):
    assert size_idea(idea(), REALISTIC, cash=cash, equity=equity, held_qty=0) == 0


def test_size_zero_position_cap_is_zero():
    book = REALISTIC.model_copy(update={"max_position_pct": 0.0})
    assert size_idea(idea(), book, cash=1000.0, equity=1000.0, held_qty=0) == 0


def test_size_returns_a_plain_int():
    assert type(size_idea(idea(), REALISTIC, cash=1000.0, equity=1000.0, held_qty=0)) is int
    assert type(size_idea(idea(Side.SELL), REALISTIC, cash=0.0, equity=1000.0, held_qty=2)) is int
