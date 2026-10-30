"""The scheduled jobs: premarket, open_session, intraday, exits_job, postmarket and weekly.

Every job is ``async`` and takes ``(svc, now)`` with ``now`` an aware IST datetime. ``app.guarded``
wraps them (trading-day check, error alert, ``last_run:<job>``). Shared helpers live in
``pipeline_support``.

State keys (all via ``store.set_state``):

* ``news_since`` (ISO datetime): news is fetched from here on; moved forward on every fetch.
* ``todays_ideas`` (list of idea ids) and ``todays_ideas_day`` (ISO date): today's PM ideas.
  open_session acts only on ideas from today.
* ``shortlist`` (symbols), ``premarket_prices`` ({symbol: price}), ``news_digest`` and
  ``technicals`` (model dumps): premarket's view, which intraday compares against and reuses.
* ``last_intraday_ai`` (ISO datetime): last intraday AI run (at most one per 2 hours when REDUCED).
* ``bench_start`` ({"day", "price"}): NIFTYBEES' close on the first postmarket, for the benchmark.
* ``fundamentals`` ({symbol: FundamentalView dump}): written weekly, kept when unavailable.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Any

from aitrader.agents.analysts import run_fundamental, run_news, run_technical
from aitrader.agents.llm import BudgetLevel, budget_level
from aitrader.agents.portfolio import run_portfolio_manager
from aitrader.agents.reviewer import run_daily_review, run_weekly_review
from aitrader.agents.schemas import NewsDigest, TechnicalBatch
from aitrader.calendar import IST, in_window
from aitrader.data.indicators import IndicatorSnapshot, compute_indicators
from aitrader.data.news import NewsItem, count_hits, fetch_news, prefilter
from aitrader.data.screener import screen
from aitrader.execution import describe_error
from aitrader.golive import evaluate_go_live, format_report
from aitrader.models import Quote
from aitrader.pipeline_support import (
    ACTED_KEY,
    BENCH,
    FUNDAMENTALS_KEY,
    act_on_ideas,
    ai_allowed,
    book_names,
    check_limits,
    fetch_quotes,
    held_symbols,
    inr,
    ist_day,
    last_close,
    load_model,
    notify,
    pm_context,
    unacted_ideas,
    usable,
    usd,
)
from aitrader.risk.equity import book_equity
from aitrader.risk.manager import MARKET_CLOSE

log = logging.getLogger(__name__)

HISTORY_DAYS = 60
REALISTIC = "realistic"
REALISTIC_SLOTS = 5               # extra shortlist places for stocks the realistic book can afford
MOVE_TRIGGER_PCT = 2.0            # intraday PM re-run when a watched symbol moved more than this
REDUCED_INTERVAL = timedelta(hours=2)
NEWS_LOOKBACK = timedelta(hours=24)
MARKET_KEYWORDS = ["Nifty", "Sensex", "NSE", "BSE", "RBI", "SEBI", "repo rate", "inflation", "FII", "DII",
                   "rupee", "crude", "GDP", "stock market", "Dalal Street"]

TODAYS_IDEAS_KEY = "todays_ideas"
TODAYS_DAY_KEY = "todays_ideas_day"
NEWS_SINCE_KEY = "news_since"
SHORTLIST_KEY = "shortlist"
PREMARKET_PRICES_KEY = "premarket_prices"
NEWS_DIGEST_KEY = "news_digest"
TECHNICALS_KEY = "technicals"
LAST_AI_KEY = "last_intraday_ai"
BENCH_START_KEY = "bench_start"


def _parse_dt(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt


def _todays_ideas(svc: Any, now: datetime) -> list:
    store = svc.store
    if store.get_state(TODAYS_DAY_KEY) != ist_day(now):
        return []
    ideas = [store.get_idea(i) for i in store.get_state(TODAYS_IDEAS_KEY) or []]
    return [i for i in ideas if i is not None]


async def _news_since_last(svc: Any, now: datetime) -> list[NewsItem]:
    since = _parse_dt(svc.store.get_state(NEWS_SINCE_KEY)) or now - NEWS_LOOKBACK
    try:
        items = await asyncio.to_thread(fetch_news, svc.settings.data.rss_feeds, since, svc.news_parse)
    except Exception as exc:  # noqa: BLE001 - news is optional input
        log.warning("news fetch failed: %s", describe_error(exc))
        return []
    svc.store.set_state(NEWS_SINCE_KEY, now.isoformat())
    return items


def _indicators(svc: Any, symbols: list[str]) -> dict[str, IndicatorSnapshot]:
    """Blocking: daily history and indicators for ``symbols`` (run in a thread)."""
    snaps: dict[str, IndicatorSnapshot] = {}
    for symbol in symbols:
        try:
            snap = compute_indicators(symbol, svc.market_data.daily_history(symbol, HISTORY_DAYS))
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the rest
            log.warning("indicators for %s failed: %s", symbol, describe_error(exc))
            continue
        if snap is not None:
            snaps[symbol] = snap
    return snaps


def _shortlist(svc: Any, snaps: dict[str, IndicatorSnapshot], hits: dict[str, int], held: set[str]) -> list[str]:
    """Held symbols plus the top scorers, then a few more the realistic book can afford."""
    shortlist = screen(snaps, hits, held, None, svc.settings.data.shortlist_size)
    if any(b.name == REALISTIC for b in svc.settings.books):
        try:
            cash = svc.store.cash(REALISTIC)
        except KeyError:
            cash = 0.0
        if cash > 0:
            shortlist += screen(snaps, hits, set(), cash, REALISTIC_SLOTS)
    return list(dict.fromkeys(shortlist))


# ---------------------------------------------------------------- premarket

async def premarket(svc: Any, now: datetime) -> None:
    store = svc.store
    bench = (await fetch_quotes(svc, [BENCH])).get(BENCH)
    try:
        history = await asyncio.to_thread(svc.market_data.daily_history, BENCH, HISTORY_DAYS)
    except Exception as exc:  # noqa: BLE001
        log.warning("daily history for %s failed: %s", BENCH, describe_error(exc))
        history = None
    # Either source is enough: before the open the 5-minute quote feed is often empty.
    if bench is None and (history is None or len(history) == 0):
        await notify(svc, f"Data feed down: no quote or daily history for {BENCH}. Skipping today's premarket "
                          "AI steps; stop/target exits and loss limits keep running.")
        with store.transaction():                      # nothing from an earlier day may be reused
            for key in (SHORTLIST_KEY, PREMARKET_PRICES_KEY, TECHNICALS_KEY, NEWS_DIGEST_KEY, TODAYS_IDEAS_KEY):
                store.set_state(key, None)
        return

    items = await _news_since_last(svc, now)
    relevant = prefilter(items, svc.aliases, MARKET_KEYWORDS)
    snaps = await asyncio.to_thread(_indicators, svc, list(svc.universe))
    held = held_symbols(svc)
    shortlist = _shortlist(svc, snaps, count_hits(relevant, svc.aliases), held)
    quotes = await fetch_quotes(svc, set(shortlist) | held | {BENCH})
    for symbol in set(shortlist) | held:              # before the open: fall back to the last daily close
        if symbol not in quotes and symbol in snaps:
            quotes[symbol] = Quote(symbol, snaps[symbol].close, now)

    with store.transaction():
        store.set_state(SHORTLIST_KEY, shortlist)
        store.set_state(PREMARKET_PRICES_KEY, {s: q.price for s, q in quotes.items()})
        store.set_state(TODAYS_IDEAS_KEY, [])
        store.set_state(TODAYS_DAY_KEY, ist_day(now))
        store.set_state(ACTED_KEY, [])
        store.set_state(NEWS_DIGEST_KEY, None)
        store.set_state(TECHNICALS_KEY, None)

    if not ai_allowed(svc, now):
        await notify(svc, "AI budget exhausted: no new ideas today. Stop/target exits and loss limits keep running.")
        return

    digest = await asyncio.to_thread(run_news, svc.llm, relevant, list(svc.universe))
    technicals = await asyncio.to_thread(run_technical, svc.llm, [snaps[s] for s in shortlist if s in snaps])
    store.set_state(NEWS_DIGEST_KEY, digest.model_dump(mode="json") if digest is not None else None)
    store.set_state(TECHNICALS_KEY, technicals.model_dump(mode="json") if technicals is not None else None)

    ctx = await pm_context(svc, now, quotes, news=digest, technicals=technicals)
    ideas = await asyncio.to_thread(run_portfolio_manager, svc.llm, store, ctx)
    store.set_state(TODAYS_IDEAS_KEY, [i.id for i in ideas])
    listed = "; ".join(f"{i.side} {i.symbol} @ {inr(i.limit_price)}" for i in ideas) or "none"
    await notify(svc, f"Premarket: {len(relevant)} relevant news item(s), {len(shortlist)} stocks shortlisted. "
                      f"Ideas for today: {listed}.")


# ---------------------------------------------------------------- open session

async def open_session(svc: Any, now: datetime) -> None:
    ideas = _todays_ideas(svc, now)
    if not ideas:
        log.info("open_session: no ideas for today")
        return
    await act_on_ideas(svc, ideas, now)


# ---------------------------------------------------------------- intraday

async def _trigger(svc: Any, now: datetime, watched: set[str]
                   ) -> tuple[str | None, NewsDigest | None, dict[str, Quote]]:
    """Why the PM should run again (or ``None``), a fresh news digest if one was made, and quotes."""
    store = svc.store
    quotes = await fetch_quotes(svc, watched)
    base = store.get_state(PREMARKET_PRICES_KEY) or {}
    for symbol in sorted(watched):
        before, quote = base.get(symbol), quotes.get(symbol)
        if usable(before) and quote is not None:
            move = (quote.price / before - 1) * 100
            if abs(move) > MOVE_TRIGGER_PCT:
                return f"{symbol} moved {move:+.1f}% since premarket", None, quotes

    items = await _news_since_last(svc, now)
    relevant = prefilter(items, {s: svc.aliases.get(s, []) for s in watched}, [])
    if not relevant:
        return None, None, quotes
    store.set_state(LAST_AI_KEY, now.isoformat())
    digest = await asyncio.to_thread(run_news, svc.llm, relevant, sorted(watched))
    if digest is not None:
        high = [s.symbol for s in digest.symbols if s.materiality == "high"]
        if high:
            return f"high-materiality news on {', '.join(high)}", digest, quotes
    return None, None, quotes


async def intraday(svc: Any, now: datetime) -> None:
    store = svc.store
    await _expire_cards(svc, now)
    pending = store.pending_orders()
    await svc.executor.retry_pending(await fetch_quotes(svc, {o.symbol for o in pending}), now)
    await check_limits(svc, now)

    level = budget_level(store, svc.settings.ai, now)
    if level is BudgetLevel.EXHAUSTED:
        log.info("intraday: AI budget exhausted, no AI re-run")
        return
    last = _parse_dt(store.get_state(LAST_AI_KEY))
    if level is BudgetLevel.REDUCED and last is not None and now - last < REDUCED_INTERVAL:
        log.info("intraday: AI budget reduced, last AI run at %s", last.isoformat(timespec="minutes"))
        return

    watched = held_symbols(svc) | set(store.get_state(SHORTLIST_KEY) or [])
    reason, digest, quotes = await _trigger(svc, now, watched)
    if reason is None:
        return
    store.set_state(LAST_AI_KEY, now.isoformat())
    # The prices this re-run saw become the new baseline, so one move triggers one re-run.
    base = dict(store.get_state(PREMARKET_PRICES_KEY) or {})
    base.update({s: q.price for s, q in quotes.items()})
    store.set_state(PREMARKET_PRICES_KEY, base)
    if digest is not None:
        store.set_state(NEWS_DIGEST_KEY, digest.model_dump(mode="json"))
    ctx = await pm_context(svc, now, quotes, news=digest or load_model(store, NEWS_DIGEST_KEY, NewsDigest),
                           technicals=load_model(store, TECHNICALS_KEY, TechnicalBatch))
    ideas = await asyncio.to_thread(run_portfolio_manager, svc.llm, store, ctx)
    known = [i.id for i in _todays_ideas(svc, now)]
    store.set_state(TODAYS_IDEAS_KEY, known + [i.id for i in ideas])
    store.set_state(TODAYS_DAY_KEY, ist_day(now))
    if ideas:
        await notify(svc, f"Intraday re-run ({reason}): {len(ideas)} new idea(s).")
        await act_on_ideas(svc, ideas, now)


# ---------------------------------------------------------------- exits

async def _expire_cards(svc: Any, now: datetime) -> None:
    expired = svc.approvals.expire_due(now)
    if expired:
        await notify(svc, f"{len(expired)} approval card(s) expired unanswered and were not placed.")


async def exits_job(svc: Any, now: datetime) -> None:
    """Stops and targets, limits and card expiry; and, inside the entry window, today's ideas that
    could not be acted on yet (no fresh quote at 09:20). No AI calls."""
    if not in_window(now, svc.settings.risk.exit_start, MARKET_CLOSE):
        return
    await svc.exits.check(now)
    await _expire_cards(svc, now)
    await check_limits(svc, now)
    risk = svc.settings.risk
    if in_window(now, risk.entry_start, risk.entry_end):
        waiting = unacted_ideas(svc, _todays_ideas(svc, now))
        if waiting:
            await act_on_ideas(svc, waiting, now)


# ---------------------------------------------------------------- postmarket

async def _closing_prices(svc: Any, now: datetime, symbols: set[str]) -> dict[str, float]:
    """Bhavcopy closes, falling back to the feed's last daily close for anything missing."""
    try:
        closes = await asyncio.to_thread(svc.fetch_closes, now.astimezone(IST).date())
    except Exception as exc:  # noqa: BLE001
        log.warning("bhavcopy failed: %s", describe_error(exc))
        closes = {}
    prices = {s: float(closes[s]) for s in symbols if usable(closes.get(s))}
    for symbol in sorted(symbols - set(prices)):
        price = await asyncio.to_thread(last_close, svc, symbol)
        if price is not None:
            prices[symbol] = price
    return prices


def _bench_line(svc: Any, prices: dict[str, float]) -> str:
    start = svc.store.get_state(BENCH_START_KEY)
    price = prices.get(BENCH)
    if not start or price is None:
        return f"{BENCH} benchmark: no close today"
    pct = (price / start["price"] - 1) * 100
    return f"{BENCH} since {start['day']}: {pct:+.2f}% ({inr(start['price'])} -> {inr(price)})"


async def postmarket(svc: Any, now: datetime) -> None:
    store, settings = svc.store, svc.settings
    today = now.astimezone(IST).date()
    prices = await _closing_prices(svc, now, held_symbols(svc) | {BENCH})
    quotes = {s: Quote(s, p, now) for s, p in prices.items()}

    equities: dict[str, float] = {}
    for name in book_names(svc):
        try:
            equities[name] = book_equity(store, name, quotes, settings.charges)
        except KeyError as exc:
            log.error("equity for %s unavailable: %s", name, describe_error(exc))
            continue
        store.save_equity_snapshot(name, today, equities[name])
    await check_limits(svc, now, quotes)
    if store.get_state(BENCH_START_KEY) is None and BENCH in prices:
        store.set_state(BENCH_START_KEY, {"day": today.isoformat(), "price": prices[BENCH]})

    review = None
    if ai_allowed(svc, now):
        review = await asyncio.to_thread(run_daily_review, svc.llm, store, today, book_names(svc),
                                         ideas=_todays_ideas(svc, now), quotes=prices)

    start = datetime.combine(today, time.min, tzinfo=IST)
    end = start + timedelta(days=1)
    lines = [f"Daily summary, {today.strftime('%a %d %b %Y')}"]
    for name, equity in equities.items():
        fills = store.fills(name, start, end)
        before = store.latest_equity_snapshot_before(name, today)
        baseline = svc.limits[name].baseline()
        day_pnl = equity - (before if before is not None else baseline)
        lines.append(f"{name}: equity {inr(equity)}; today {inr(day_pnl, True)}; since baseline {inr(baseline)}: "
                     f"{inr(equity - baseline, True)}; {len(fills)} trade(s), charges paid "
                     f"{inr(sum(f.charges for f in fills))}")
    ai, local = settings.ai, now.astimezone(IST)
    today_usd = store.ai_spend_usd_on(today)
    month_usd = store.ai_spend_usd(local.year, local.month)
    lines.append(f"AI spend (a learning cost, apart from trading P&L): today {usd(today_usd)} = "
                 f"{inr(today_usd * ai.usd_inr)}; this month {usd(month_usd)} = {inr(month_usd * ai.usd_inr)} "
                 f"of the {usd(ai.monthly_budget_usd)} cap")
    lines.append(_bench_line(svc, prices))
    if review is not None and review.lessons:
        lines.append("Lessons added:")
        lines += [f"- {lesson}" for lesson in review.lessons]
    else:
        lines.append("Lessons added: none" + ("" if ai_allowed(svc, now) else " (AI budget exhausted)"))
    await notify(svc, "\n".join(lines))


# ---------------------------------------------------------------- weekly

async def bench_return_pct(svc: Any) -> float:
    """NIFTYBEES' return since ``bench_start`` in percent, or NaN when unknown."""
    start = svc.store.get_state(BENCH_START_KEY)
    if not start:
        return float("nan")
    quote = (await fetch_quotes(svc, [BENCH])).get(BENCH)
    price = quote.price if quote is not None else await asyncio.to_thread(last_close, svc, BENCH)
    if price is None:
        return float("nan")
    return (price / start["price"] - 1) * 100


def _fundamentals_data(svc: Any) -> dict[str, dict]:
    """Blocking: non-empty fundamentals for the universe (run in a thread)."""
    data: dict[str, dict] = {}
    for symbol in svc.universe:
        try:
            fields = svc.market_data.fundamentals(symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("fundamentals for %s failed: %s", symbol, describe_error(exc))
            continue
        if fields:
            data[symbol] = fields
    return data


async def weekly(svc: Any, now: datetime) -> None:
    store = svc.store
    today = now.astimezone(IST).date()
    lines = [f"Weekly report, week to {today.strftime('%d %b %Y')}"]
    if not ai_allowed(svc, now):
        lines.append("AI budget exhausted: no fundamentals or weekly review this week.")
    else:
        data = await asyncio.to_thread(_fundamentals_data, svc)
        batch = await asyncio.to_thread(run_fundamental, svc.llm, data) if data else None
        if batch is None or not batch.views:
            lines.append("Fundamentals unavailable this week; last week's are kept.")
        else:
            store.set_state(FUNDAMENTALS_KEY, {v.symbol: v.model_dump(mode="json") for v in batch.views})
            lines.append(f"Fundamentals updated for {len(batch.views)} stock(s).")
        notes = await asyncio.to_thread(run_weekly_review, svc.llm, store, today, books=book_names(svc))
        lines.append(f"Weekly review:\n{notes}" if notes else "Weekly review: no trades this week.")
    report = evaluate_go_live(store, svc.settings, now, await bench_return_pct(svc))
    lines.append(format_report(report))
    await notify(svc, "\n".join(lines))
