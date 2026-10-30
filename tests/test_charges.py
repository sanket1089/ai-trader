import pytest

from aitrader.charges import ChargeBreakdown, compute_charges, round_trip_charges
from aitrader.config import ChargesConfig
from aitrader.models import Side

CFG = ChargesConfig(brokerage_per_order=0.0, stt_pct=0.1, stamp_buy_pct=0.015,
                    exchange_pct=0.00322, sebi_pct=0.0001, gst_pct=18.0, dp_per_sell=15.34)


def test_buy_charges_niftybees():
    c = compute_charges(Side.BUY, 3, 280.0, CFG)          # turnover 840
    assert c.stt == pytest.approx(0.84)
    assert c.stamp == pytest.approx(0.126)
    assert c.dp == 0
    assert c.total == pytest.approx(0.99890784, abs=1e-6)


def test_sell_charges_include_dp():
    c = compute_charges(Side.SELL, 3, 300.0, CFG)         # turnover 900
    assert c.stamp == 0
    assert c.dp == 15.34
    assert c.total == pytest.approx(16.2752584, abs=1e-6)


def test_round_trip():
    assert round_trip_charges(3, 280.0, 300.0, CFG) == pytest.approx(0.99890784 + 16.2752584, abs=1e-6)


def test_gst_applies_to_brokerage_exchange_and_sebi_only():
    cfg = CFG.model_copy(update={"brokerage_per_order": 20.0})
    c = compute_charges(Side.BUY, 10, 100.0, cfg)         # turnover 1000
    assert c.brokerage == 20.0
    assert c.exchange == pytest.approx(0.0322)
    assert c.sebi == pytest.approx(0.001)
    assert c.gst == pytest.approx(0.18 * (20.0 + 0.0322 + 0.001))


def test_total_is_sum_of_components():
    c = ChargeBreakdown(brokerage=1, stt=2, exchange=3, sebi=4, stamp=5, gst=6, dp=7)
    assert c.total == 28


@pytest.mark.parametrize("qty, price", [(0, 100.0), (-1, 100.0), (1, 0.0), (1, -5.0)])
def test_rejects_non_positive_qty_or_price(qty, price):
    with pytest.raises(ValueError):
        compute_charges(Side.BUY, qty, price, CFG)
