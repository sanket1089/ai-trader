import math

import pytest

from aitrader.charges import compute_charges
from aitrader.config import BookConfig, ChargesConfig
from aitrader.models import Side, TradeIdea
from aitrader.sizing import size_idea

# The values in config.yaml.
CFG = ChargesConfig(brokerage_per_order=0.0, stt_pct=0.1, stamp_buy_pct=0.015, exchange_pct=0.00322,
                    sebi_pct=0.0001, gst_pct=18.0, dp_per_sell=15.34)
REALISTIC = BookConfig(name="realistic", capital=1000.0, daily_loss_limit=100.0, total_loss_limit=500.0,
                       max_entries_per_day=1, max_sell_days_per_week=3, max_position_pct=100.0)
LEARNING = BookConfig(name="learning", capital=100000.0, daily_loss_limit=10000.0,
                      total_loss_limit=50000.0, max_entries_per_day=3, max_sell_days_per_week=None,
                      max_position_pct=20.0)


def idea(side=Side.BUY, limit_price=280.0):
    return TradeIdea(id="i1", symbol="NIFTYBEES", side=side, limit_price=limit_price, stop=270.0,
                     target=300.0, expected_move_pct=3.5, confidence=0.7, holding_days=5, reason="trend")


def size(book, cash, equity, limit_price=280.0, side=Side.BUY, held_qty=0, cfg=CFG):
    return size_idea(idea(side, limit_price), book, cash, equity, held_qty, cfg)


def cost(qty, price, cfg=CFG):
    """What PaperBroker needs in cash for a BUY: value plus charges."""
    return qty * price + compute_charges(Side.BUY, qty, price, cfg).total


def test_size_realistic_takes_all_cash():
    assert size(REALISTIC, 1000.0, 1000.0) == 3  # 840 + about 1.0 of charges fits


def test_size_learning_caps_20pct():
    assert size(LEARNING, 100000.0, 100000.0) == 71  # 71 * 280 = 19880, plus about 19.9, is under 20000


def test_size_learning_is_limited_by_cash_when_cash_is_below_the_cap():
    assert size(LEARNING, 5000.0, 100000.0) == 17  # 4760 + about 5.7 charges


def test_size_zero_when_unaffordable():
    assert size(REALISTIC, 1000.0, 1000.0, limit_price=1500.0) == 0
    assert size(LEARNING, 100.0, 100000.0) == 0


def test_size_leaves_room_for_buy_charges():
    # Four units at 250 cost exactly 1000 before charges, so charges push the fourth unit out.
    assert size(REALISTIC, 1000.0, 1000.0, limit_price=250.0) == 3
    assert cost(3, 250.0) <= 1000.0 < cost(4, 250.0)
    # The reviewer's example: floor(1000 / 333) = 3, but 999 plus about 1.2 of charges is over 1000.
    assert size(REALISTIC, 1000.0, 1000.0, limit_price=333.0) == 2
    assert cost(3, 333.0) > 1000.0 >= cost(2, 333.0)


def test_size_exact_affordability_boundary():
    assert size(REALISTIC, 1000.0, 1000.0, limit_price=250.01) == 3
    assert size(REALISTIC, 1000.0, 1000.0, limit_price=332.9) == 3   # 998.7 + about 1.2 just fits
    assert cost(3, 332.9) <= 1000.0 < cost(3, 333.0)


def test_size_is_the_largest_quantity_whose_cost_fits_the_budget():
    for book, cash, equity in [(REALISTIC, 1000.0, 1000.0), (REALISTIC, 777.77, 800.0),
                               (LEARNING, 100000.0, 100000.0), (LEARNING, 3000.0, 90000.0)]:
        budget = min(cash, book.max_position_pct * equity / 100)
        for price in [1.0, 9.95, 60.0, 123.45, 250.0, 280.0, 333.0, 499.99, 999.0]:
            qty = size(book, cash, equity, limit_price=price)
            assert qty == 0 or cost(qty, price) <= budget, (book.name, cash, price, qty)
            assert cost(qty + 1, price) > budget, (book.name, cash, price, qty)


def test_size_flat_brokerage_reduces_quantity_and_can_zero_it():
    cfg = CFG.model_copy(update={"brokerage_per_order": 200.0})   # 236 with GST, per order
    assert size(REALISTIC, 1000.0, 1000.0, cfg=cfg) == 2          # 560 + 236 + about 0.7
    assert size(REALISTIC, 1000.0, 1000.0, limit_price=1.0, cfg=cfg) == 763
    cfg = CFG.model_copy(update={"brokerage_per_order": 2000.0})
    assert size(REALISTIC, 1000.0, 1000.0, limit_price=1.0, cfg=cfg) == 0


def test_size_never_spends_more_than_the_budget_despite_float_error():
    # 1.1 / 0.1 is 11.000000000000002 in floating point, but 11 units cost 1.1000000000000001 > 1.1.
    qty = size(REALISTIC, 1.1, 1.1, limit_price=0.1)
    assert cost(qty, 0.1) <= 1.1


def test_size_sell_returns_held_qty():
    assert size(LEARNING, 0.0, 100000.0, side=Side.SELL, held_qty=7) == 7
    assert size(REALISTIC, 0.0, 1000.0, side=Side.SELL, held_qty=0) == 0


def test_size_sell_ignores_charges_and_cash():
    assert size(REALISTIC, 0.0, 0.0, limit_price=3500.0, side=Side.SELL, held_qty=2) == 2


def test_size_sell_never_negative():
    assert size(LEARNING, 0.0, 100000.0, side=Side.SELL, held_qty=-3) == 0


@pytest.mark.parametrize("price", [0.0, -5.0, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_size_invalid_limit_price_is_zero(price, side):
    assert size(REALISTIC, 1000.0, 1000.0, limit_price=price, side=side, held_qty=5) == 0


@pytest.mark.parametrize("cash,equity", [(-100.0, 1000.0), (math.nan, 1000.0), (1000.0, math.nan),
                                         (math.inf, 1000.0), (1000.0, math.inf), (1000.0, -1000.0),
                                         (0.0, 1000.0)])
def test_size_buy_with_bad_money_is_zero(cash, equity):
    assert size(REALISTIC, cash, equity) == 0


def test_size_zero_position_cap_is_zero():
    book = REALISTIC.model_copy(update={"max_position_pct": 0.0})
    assert size(book, 1000.0, 1000.0) == 0


def test_size_returns_a_plain_int():
    assert type(size(REALISTIC, 1000.0, 1000.0)) is int
    assert type(size(REALISTIC, 0.0, 1000.0, side=Side.SELL, held_qty=2)) is int
