"""Delivery (CNC) charge model for NSE cash equities and ETFs.

All rates in ``ChargesConfig`` are percentages of turnover (qty x price), except
the flat INR amounts ``brokerage_per_order`` and ``dp_per_sell``.
"""

from dataclasses import dataclass

from aitrader.config import ChargesConfig
from aitrader.models import Side


@dataclass(frozen=True)
class ChargeBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float
    dp: float

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt + self.exchange + self.sebi
                + self.stamp + self.gst + self.dp)


def compute_charges(side: Side, qty: int, price: float, cfg: ChargesConfig) -> ChargeBreakdown:
    """Charges for one executed order. Stamp duty applies to buys, the DP charge to sells."""
    if qty < 1:
        raise ValueError(f"qty must be a whole number >= 1, got {qty}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    turnover = qty * price
    brokerage = cfg.brokerage_per_order
    exchange = turnover * cfg.exchange_pct / 100
    sebi = turnover * cfg.sebi_pct / 100
    return ChargeBreakdown(
        brokerage=brokerage,
        stt=turnover * cfg.stt_pct / 100,
        exchange=exchange,
        sebi=sebi,
        stamp=turnover * cfg.stamp_buy_pct / 100 if side is Side.BUY else 0.0,
        gst=(brokerage + exchange + sebi) * cfg.gst_pct / 100,
        dp=cfg.dp_per_sell if side is Side.SELL else 0.0,
    )


def round_trip_charges(qty: int, buy_price: float, sell_price: float, cfg: ChargesConfig) -> float:
    """Total charges for buying then selling ``qty`` units."""
    return (compute_charges(Side.BUY, qty, buy_price, cfg).total
            + compute_charges(Side.SELL, qty, sell_price, cfg).total)
