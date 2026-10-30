"""Output schemas for every agent.

These are sent to Claude as structured-output JSON schemas (``anthropic.transform_schema``), so
they stay plain: no validators and no numeric ``Field`` constraints. Ranges are stated in field
descriptions for the model and enforced afterwards in code (see ``analysts``).
"""

from typing import Literal

from pydantic import BaseModel, Field


class SymbolNews(BaseModel):
    symbol: str
    sentiment: float = Field(description="-1 (very negative) to 1 (very positive)")
    materiality: Literal["low", "medium", "high"]
    summary: str


class NewsDigest(BaseModel):
    market_mood: Literal["bullish", "neutral", "bearish"]
    market_summary: str
    symbols: list[SymbolNews]


class TechnicalView(BaseModel):
    symbol: str
    trend: Literal["up", "down", "sideways"]
    support: float = Field(description="support price in rupees")
    resistance: float = Field(description="resistance price in rupees")
    signal: Literal["buy", "hold", "sell", "avoid"]
    confidence: float = Field(description="0 to 1")
    notes: str


class TechnicalBatch(BaseModel):
    views: list[TechnicalView]


class FundamentalView(BaseModel):
    symbol: str
    quality: Literal["strong", "ok", "weak"]
    valuation: Literal["cheap", "fair", "expensive"]
    notes: str


class FundamentalBatch(BaseModel):
    views: list[FundamentalView]


class IdeaOut(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    limit_price: float
    stop: float
    target: float
    expected_move_pct: float = Field(description="expected move to target, in percent")
    confidence: float = Field(description="0 to 1")
    holding_days: int
    reason: str


class PMDecision(BaseModel):
    market_view: str
    ideas: list[IdeaOut]


class RecoveryPlan(BaseModel):
    what_went_wrong: str
    changes: list[str]
    mode: Literal["reduced_size", "exit_only", "normal"]
    next_trades: list[IdeaOut]


class TradeReview(BaseModel):
    symbol: str
    verdict: Literal["good", "bad", "neutral"]
    note: str


class ReviewOutput(BaseModel):
    lessons: list[str]
    trade_reviews: list[TradeReview]


class WeeklyNotes(BaseModel):
    notes: str
