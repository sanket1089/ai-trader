"""The three analyst agents: news, technical and fundamental.

Each agent has a module-level system prompt (static, so the prompt cache can reuse it) and gets
all its data in the user message as compact JSON. The model's output is checked in code before it
is returned: symbols that were not in the input are dropped (and logged), and ``sentiment`` and
``confidence`` are clamped into range, because the schemas carry no numeric constraints.
Each ``run_*`` returns ``None`` when there is nothing to analyse or the LLM returned nothing.
"""

import dataclasses
import json
import logging
import math
from collections.abc import Collection

from aitrader.agents.llm import LLMClient
from aitrader.agents.schemas import (
    FundamentalBatch,
    FundamentalView,
    NewsDigest,
    TechnicalBatch,
)
from aitrader.data.indicators import IndicatorSnapshot
from aitrader.data.news import NewsItem

log = logging.getLogger(__name__)

MAX_NEWS_ITEMS = 150       # newest first; bounds the cost of one news call
SUMMARY_MAX_CHARS = 300    # per item, same reason
FUNDAMENTAL_BATCH = 25     # symbols per fundamental call

_MANDATE = (
    "You work for a paper-trading desk on the Indian NSE cash market only. Trades are delivery "
    "swing trades held for days to weeks; there is no F&O, no intraday and no short selling. "
    "Use only the data given in the user message, not outside knowledge of prices, results or "
    "events. When the data is thin or unclear, say \"avoid\" or \"neutral\" instead of guessing. "
    "Reply with the requested JSON only.\n\n"
)

NEWS_SYSTEM = _MANDATE + (
    "You are the news analyst. The user message is JSON with `universe` (the NSE symbols you may "
    "report on) and `news` (recent items, newest first: title, summary, source, published).\n"
    "- market_mood: bullish, neutral or bearish for Indian equities overall; market_summary: one "
    "or two sentences.\n"
    "- symbols: one entry for each universe symbol the news clearly affects; leave out symbols "
    "with no news. Never output a symbol outside the universe.\n"
    "- sentiment: -1 (very negative) to 1 (very positive), 0 if mixed. materiality: high only for "
    "news that can move the price over days to weeks (earnings, large orders, regulatory or "
    "ownership changes); low for routine or passing mentions.\n"
    "- summary: one short sentence."
)

TECHNICAL_SYSTEM = _MANDATE + (
    "You are the technical analyst. The user message is JSON with `stocks`: daily-bar indicators "
    "for a shortlist (close, sma20, sma50, rsi14, atr14, vol_ratio20 = volume vs its 20-day "
    "average, pct_change_5d in percent, high_20d, low_20d). Give exactly one view per stock and "
    "never a symbol that was not given.\n"
    "- trend: up, down or sideways. support and resistance: rupee prices near the close, taken "
    "from the 20-day range, the moving averages and ATR.\n"
    "- signal: buy, hold, sell or avoid. sell means exit an existing long (shorting is not "
    "available). Use avoid when the setup is unclear or the move already looks stretched.\n"
    "- confidence: 0 to 1, conservative. notes: one short sentence."
)

FUNDAMENTAL_SYSTEM = _MANDATE + (
    "You are the fundamental analyst. The user message is JSON with `fundamentals`: a map from "
    "NSE symbol to a few Yahoo Finance fields (trailingPE, priceToBook, returnOnEquity, "
    "debtToEquity, marketCap, earningsGrowth, sector); null means unknown. Give exactly one view "
    "per symbol and never a symbol that was not given.\n"
    "- quality: strong, ok or weak, from returns, leverage and growth. valuation: cheap, fair or "
    "expensive, from the multiples against growth and the sector.\n"
    "- When most fields are null or the data cannot support a judgement, stay neutral: quality "
    "\"ok\", valuation \"fair\", and say in notes that the data is thin. notes: one short sentence."
)

def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)


def _clamp(x: float, lo: float, hi: float) -> float:
    """``x`` limited to [lo, hi]; a non-finite value becomes 0 (neutral / no confidence)."""
    return min(max(x, lo), hi) if math.isfinite(x) else 0.0


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _finite_or_none(value):
    """Fundamentals can carry NaN, which would make the JSON invalid; treat it as unknown."""
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _keep_known(agent: str, items: list, allowed: Collection[str]) -> list:
    """The items whose symbol is in ``allowed``; the rest are logged and dropped."""
    kept = [i for i in items if i.symbol in allowed]
    if len(kept) < len(items):
        dropped = [i.symbol for i in items if i.symbol not in allowed]
        log.info("agent=%s dropped %d view(s) for symbols not in the input: %s",
                 agent, len(dropped), ", ".join(dropped))
    return kept


def run_news(llm: LLMClient, items: list[NewsItem], universe: list[str]) -> NewsDigest | None:
    if not items:
        log.info("agent=news no news items, skipping the LLM call")
        return None
    newest = sorted(items, key=lambda i: i.published, reverse=True)[:MAX_NEWS_ITEMS]
    user = _dumps({
        "universe": list(universe),
        "news": [{"title": i.title, "summary": _clip(i.summary, SUMMARY_MAX_CHARS),
                  "source": i.source, "published": i.published.isoformat(timespec="minutes")}
                 for i in newest],
    })
    digest = llm.run("news", NEWS_SYSTEM, user, NewsDigest)
    if digest is None:
        return None
    symbols = [s.model_copy(update={"sentiment": _clamp(s.sentiment, -1.0, 1.0)})
               for s in _keep_known("news", digest.symbols, set(universe))]
    return digest.model_copy(update={"symbols": symbols})


def run_technical(llm: LLMClient, snaps: list[IndicatorSnapshot]) -> TechnicalBatch | None:
    if not snaps:
        log.info("agent=technical no snapshots, skipping the LLM call")
        return None
    stocks = [{k: round(v, 2) if isinstance(v, float) else v
               for k, v in dataclasses.asdict(s).items()} for s in snaps]
    batch = llm.run("technical", TECHNICAL_SYSTEM, _dumps({"stocks": stocks}), TechnicalBatch)
    if batch is None:
        return None
    views = [v.model_copy(update={"confidence": _clamp(v.confidence, 0.0, 1.0)})
             for v in _keep_known("technical", batch.views, {s.symbol for s in snaps})]
    return TechnicalBatch(views=views)


def run_fundamental(llm: LLMClient, data: dict[str, dict]) -> FundamentalBatch | None:
    """Weekly, in batches of 25. Symbols with no fundamentals are not sent; the views from all
    batches are merged, and a batch that returns nothing is skipped (``None`` if all do)."""
    usable = {}
    for symbol, fields in data.items():
        cleaned = {k: _finite_or_none(v) for k, v in (fields or {}).items()}
        if any(v is not None for v in cleaned.values()):
            usable[symbol] = cleaned
    skipped = [s for s in data if s not in usable]
    if skipped:
        log.info("agent=fundamental skipping %d symbol(s) with no fundamentals: %s",
                 len(skipped), ", ".join(skipped))
    if not usable:
        return None

    symbols = list(usable)
    views: list[FundamentalView] = []
    got_any = False
    for start in range(0, len(symbols), FUNDAMENTAL_BATCH):
        chunk = symbols[start:start + FUNDAMENTAL_BATCH]
        user = _dumps({"fundamentals": {s: usable[s] for s in chunk}})
        batch = llm.run("fundamental", FUNDAMENTAL_SYSTEM, user, FundamentalBatch)
        if batch is None:
            log.warning("agent=fundamental batch of %d symbol(s) returned nothing: %s",
                        len(chunk), ", ".join(chunk))
            continue
        got_any = True
        views.extend(_keep_known("fundamental", batch.views, set(chunk)))
    return FundamentalBatch(views=views) if got_any else None
