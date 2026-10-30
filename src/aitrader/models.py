"""Domain types shared across the service.

Every datetime is timezone-aware (IST, ``ZoneInfo("Asia/Kolkata")``).
Quantities are whole units; prices and charges are INR.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    ts: datetime


@dataclass
class Order:
    id: str
    book: str
    symbol: str
    side: Side
    qty: int
    limit_price: float
    reason: str
    is_exit: bool
    idea_id: str | None
    created_at: datetime
    status: OrderStatus = OrderStatus.PENDING
    attempts: int = 0


@dataclass
class Fill:
    order_id: str
    book: str
    symbol: str
    side: Side
    qty: int
    price: float
    charges: float
    is_exit: bool
    ts: datetime


@dataclass
class Position:
    book: str
    symbol: str
    qty: int
    avg_price: float
    stop: float
    target: float
    opened_at: datetime
    idea_id: str | None = None


@dataclass
class TradeIdea:
    id: str
    symbol: str
    side: Side
    limit_price: float
    stop: float
    target: float
    expected_move_pct: float
    confidence: float
    holding_days: int
    reason: str
