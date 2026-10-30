"""Helpers for the scheduled jobs in ``pipeline.py``: quotes, loss-limit evaluation and its alerts,
the once-per-stop recovery plan, the PM context, and turning ideas into approval cards or orders.

Everything blocking (the market-data feed, LLM calls) runs in a thread so the bot stays responsive.
A notifier failure is logged (error type only) and never fails a job.

State keys written here:

* ``acted_ideas`` (list of idea ids): ideas already carded or submitted today (reset by premarket),
  so no idea, and no second idea for the same (symbol, side), is acted on twice in a day. An idea
  is marked only when its card or order goes out; one skipped for a missing or stale quote waits
  for the next pass (``exits_job`` retries inside the entry window).
* ``recovery_error:<book>:<stop_id>`` and the guard's ``job_error:<job>:<type>:<date>``: once-only alerts.
* ``recovery_plan:<book>:<stop_id>`` (text) and ``recovery_sent:<book>:<stop_id>`` (bool): the
  recovery plan is written once per stop and marked sent once the notifier took it.
"""

import asyncio
import logging
import math
import uuid
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime
from typing import Any

from aitrader.agents.llm import BudgetLevel, budget_level
from aitrader.agents.portfolio import PMContext, run_recovery_plan
from aitrader.agents.schemas import FundamentalView, NewsDigest, RecoveryPlan, TechnicalBatch
from aitrader.approvals import get_autopilot, set_autopilot
from aitrader.calendar import IST
from aitrader.data.market_data import is_stale
from aitrader.execution import describe_error, marketable_limit
from aitrader.models import Order, Quote, Side, TradeIdea
from aitrader.risk.equity import book_equity
from aitrader.risk.limits import LimitEval, LimitState
from aitrader.risk.manager import MANUAL_STOP_KEY, PAUSED_KEY
from aitrader.sizing import size_idea

log = logging.getLogger(__name__)

BENCH = "NIFTYBEES"
LEARNING = "learning"
LESSONS_FOR_PM = 10
ACTED_KEY = "acted_ideas"
FUNDAMENTALS_KEY = "fundamentals"


# ---------------------------------------------------------------- small helpers

def inr(x: float, signed: bool = False) -> str:
    x = round(x, 2) + 0.0
    if signed:
        return f"{'+' if x >= 0 else '-'}₹{abs(x):,.2f}"
    return f"₹{x:,.2f}"


def usd(x: float) -> str:
    return f"US${x:,.2f}"


def usable(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def book_names(svc: Any) -> list[str]:
    return [b.name for b in svc.settings.books]


def held_symbols(svc: Any) -> set[str]:
    return {p.symbol for name in book_names(svc) for p in svc.store.positions(name)}


def job_lock(svc: Any, name: str) -> asyncio.Lock:
    """A lock per ``svc`` and ``name``: jobs run as concurrent tasks (exits every 3 minutes overlaps
    intraday), and a read-check-write across an ``await`` must not run twice at once."""
    locks = svc.__dict__.setdefault("_job_locks", {})
    return locks.setdefault(name, asyncio.Lock())


def ai_allowed(svc: Any, now: datetime) -> bool:
    return budget_level(svc.store, svc.settings.ai, now) is not BudgetLevel.EXHAUSTED


async def notify(svc: Any, text: str) -> bool:
    try:
        await svc.notifier.send(text)
        return True
    except Exception as exc:  # noqa: BLE001 - a down notifier must never fail a job
        log.error("notifier failed: %s", describe_error(exc))
        return False


async def notify_once(svc: Any, key: str, text: str) -> bool:
    """Send once per state ``key`` (through ``executor.notify_once`` when it has one); the key is set
    only after the message went through."""
    sender = getattr(svc.executor, "notify_once", None)
    if callable(sender):
        return bool(await sender(key, text))
    if svc.store.get_state(key):
        return False
    if await notify(svc, text):
        svc.store.set_state(key, True)
        return True
    return False


async def fetch_quotes(svc: Any, symbols: Iterable[str]) -> dict[str, Quote]:
    """Usable quotes for ``symbols`` (the feed runs in a thread). A feed failure gives ``{}``."""
    wanted = sorted(set(symbols))
    if not wanted:
        return {}
    try:
        quotes = await asyncio.to_thread(svc.market_data.quotes, wanted)
    except Exception as exc:  # noqa: BLE001
        log.warning("quotes unavailable: %s", describe_error(exc))
        return {}
    return {s: q for s, q in dict(quotes).items() if q is not None and usable(q.price)}


def last_close(svc: Any, symbol: str, days: int = 5) -> float | None:
    """The last daily close from the market-data feed (blocking), or ``None``."""
    try:
        df = svc.market_data.daily_history(symbol, days)
        closes = [float(c) for c in df["close"].tolist()] if len(df) else []
    except Exception as exc:  # noqa: BLE001
        log.warning("daily history unavailable for %s: %s", symbol, describe_error(exc))
        return None
    closes = [c for c in closes if usable(c)]
    return closes[-1] if closes else None


def load_fundamentals(store: Any) -> dict[str, FundamentalView]:
    out: dict[str, FundamentalView] = {}
    for symbol, raw in (store.get_state(FUNDAMENTALS_KEY) or {}).items():
        try:
            out[symbol] = FundamentalView.model_validate(raw)
        except Exception:  # noqa: BLE001 - one bad row must not lose the rest
            log.warning("ignoring a stored fundamental view for %s that no longer validates", symbol)
    return out


def load_model(store: Any, key: str, model: type) -> Any:
    raw = store.get_state(key)
    if raw is None:
        return None
    try:
        return model.model_validate(raw)
    except Exception:  # noqa: BLE001
        log.warning("ignoring stored %s that no longer validates", key)
        return None


async def pm_context(svc: Any, now: datetime, quotes: dict[str, Quote], *, book: str = LEARNING,
                     news: NewsDigest | None = None, technicals: TechnicalBatch | None = None) -> PMContext:
    try:
        index = await asyncio.to_thread(svc.market_data.index_snapshot)
    except Exception as exc:  # noqa: BLE001
        log.warning("index snapshot unavailable: %s", describe_error(exc))
        index = {}
    cash = {}
    for name in book_names(svc):
        try:
            cash[name] = svc.store.cash(name)
        except KeyError:
            continue
    return PMContext(now=now, news=news, technicals=technicals, fundamentals=load_fundamentals(svc.store),
                     index=dict(index or {}), positions=svc.store.positions(book), cash=cash,
                     lessons=svc.store.recent_lessons(LESSONS_FOR_PM),
                     quotes={s: q.price for s, q in quotes.items()})


# ---------------------------------------------------------------- loss limits

async def check_limits(svc: Any, now: datetime, quotes: dict[str, Quote] | None = None) -> dict[str, LimitEval]:
    """Evaluate every book's loss limits at current equity and act on the transitions.

    A newly triggered hard stop sets autopilot OFF and alerts the owner; a newly triggered daily
    pause alerts once. While a book is hard stopped its recovery plan is sent once per stop.
    """
    if quotes is None:
        quotes = await fetch_quotes(svc, held_symbols(svc))
    out: dict[str, LimitEval] = {}
    for cfg in svc.settings.books:
        limits = svc.limits[cfg.name]
        try:
            equity = book_equity(svc.store, cfg.name, quotes, svc.settings.charges)
            ev = limits.evaluate(equity, now)
        except (KeyError, ValueError) as exc:
            log.error("limit check for %s failed: %s", cfg.name, describe_error(exc))
            continue
        out[cfg.name] = ev
        if ev.newly_triggered and ev.state is LimitState.HARD_STOPPED:
            set_autopilot(svc.store, False)
            await notify(svc, f"HARD STOP on {cfg.name}: total loss {inr(ev.total_loss)} reached the "
                              f"{inr(cfg.total_loss_limit)} limit. No new entries in this book and autopilot "
                              "is OFF. Holdings are kept (never force-sold); stop and target exits keep working.")
        elif ev.newly_triggered and ev.state is LimitState.DAILY_PAUSED:
            await notify(svc, f"{cfg.name}: daily loss limit reached (down {inr(ev.daily_loss)} since the last "
                              f"close, limit {inr(cfg.daily_loss_limit)}). No new entries in this book until "
                              "the next trading day; exits keep working.")
        if ev.state is LimitState.HARD_STOPPED:
            try:
                await send_recovery_once(svc, cfg.name, ev, now, quotes)
            except Exception as exc:  # noqa: BLE001 - must never abort the entries pass that called us
                where = describe_error(exc)
                log.error("recovery plan for %s failed: %s", cfg.name, where)
                await notify_once(svc, f"recovery_error:{cfg.name}:{limits.stop_id()}",
                                  f"{cfg.name}: the recovery plan could not be written ({where}); "
                                  "it will be tried again. See the log.")
    return out


def _render_recovery(book: str, ev: LimitEval, plan: RecoveryPlan | None) -> str:
    head = f"Recovery plan for {book} (hard stop at a total loss of {inr(ev.total_loss)})"
    if plan is None:
        return "\n".join([head, "No AI recovery plan this time (AI budget exhausted or no usable reply).",
                          "Tap ✅ to approve restarting anyway, then send /resume; the baseline resets "
                          "to the current equity. Or leave the book stopped: exits keep working."])
    lines = [head, f"What went wrong: {plan.what_went_wrong}", "Changes:"]
    lines += [f"- {c}" for c in plan.changes] or ["- none"]
    lines.append(f"Mode: {plan.mode}")
    if plan.next_trades:
        lines.append("Next trades it would make:")
        lines += [f"- {t.side} {t.symbol} @ {inr(t.limit_price)} (stop {inr(t.stop)}, target "
                  f"{inr(t.target)}): {t.reason}" for t in plan.next_trades]
    lines.append("Tap ✅ to approve, then send /resume; the baseline resets to the current equity.")
    return "\n".join(lines)


async def send_recovery_once(svc: Any, book: str, ev: LimitEval, now: datetime,
                             quotes: dict[str, Quote]) -> None:
    async with job_lock(svc, "recovery"):
        await _send_recovery_once(svc, book, ev, now, quotes)


async def _send_recovery_once(svc: Any, book: str, ev: LimitEval, now: datetime,
                              quotes: dict[str, Quote]) -> None:
    store = svc.store
    stop_id = svc.limits[book].stop_id()
    sent_key = f"recovery_sent:{book}:{stop_id}"
    if store.get_state(sent_key) is True:
        return
    text_key = f"recovery_plan:{book}:{stop_id}"
    text = store.get_state(text_key)
    if text is None:                                      # the plan is written once per stop
        plan = None
        if ai_allowed(svc, now):
            ctx = await pm_context(svc, now, quotes, book=book)
            limit = next(b.total_loss_limit for b in svc.settings.books if b.name == book)
            reason = (f"The {book} book hit its total loss limit: total loss {inr(ev.total_loss)} "
                      f"against a limit of {inr(limit)}.")
            plan = await asyncio.to_thread(run_recovery_plan, svc.llm, ctx, reason)
        text = _render_recovery(book, ev, plan)
        store.set_state(text_key, text)
    sender = getattr(svc.notifier, "send_recovery", None)
    try:
        if callable(sender):
            await sender(book, stop_id, text)
        else:
            await svc.notifier.send(text)
    except Exception as exc:  # noqa: BLE001 - retried on the next limit check
        log.error("sending the recovery plan failed: %s", describe_error(exc))
        return
    store.set_state(sent_key, True)


# ---------------------------------------------------------------- ideas -> cards or orders

def _sizes(svc: Any, idea: TradeIdea, evals: dict[str, LimitEval], quotes: dict[str, Quote],
           now: datetime) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for cfg in svc.settings.books:
        if idea.side is Side.BUY:
            state = evals[cfg.name].state if cfg.name in evals else svc.limits[cfg.name].state(now)
            if state is not LimitState.ACTIVE:
                continue                                  # paused or stopped books take no entries
        try:
            cash = svc.store.cash(cfg.name)
            equity = book_equity(svc.store, cfg.name, quotes, svc.settings.charges)
        except KeyError:
            continue
        held = next((p.qty for p in svc.store.positions(cfg.name) if p.symbol == idea.symbol), 0)
        qty = size_idea(idea, cfg, cash, equity, held, svc.settings.charges)
        if qty > 0:
            sizes[cfg.name] = qty
    return sizes


def reanchor(svc: Any, idea: TradeIdea, quote: Quote) -> TradeIdea | None:
    """The idea re-priced at the fresh quote, saved under the same id, or ``None`` to skip it.

    Premarket limits are set against the previous close and may be outside the entry band by the
    time the idea is acted on. The new limit is a marketable one (``marketable_limit``): the edge of
    the entry band, because the paper broker adds slippage to the quote and a limit equal to the
    quote would never fill. BUY: stop and target kept, ``stop < limit < target`` must still hold, and
    ``expected_move_pct`` is capped at the new limit-to-target distance. SELL: only for a symbol some
    book holds. The adjusted idea is persisted so the card, the sizing, the risk cost rule and the
    review all see one version.
    """
    band = svc.settings.risk.entry_band_pct
    if idea.side is Side.BUY:
        limit = marketable_limit(Side.BUY, quote.price, band)
        if not idea.stop < limit < idea.target:
            log.info("%s: price already beyond stop/target at open (price %.2f, limit %.2f, stop %.2f, "
                     "target %.2f); idea skipped", idea.symbol, quote.price, limit, idea.stop, idea.target)
            return None
        move = min(idea.expected_move_pct, (idea.target - limit) / limit * 100)
        adjusted = replace(idea, limit_price=limit, expected_move_pct=move)
    elif idea.side is Side.SELL:
        if idea.symbol not in held_symbols(svc):
            log.info("%s: SELL idea skipped, the symbol is not held", idea.symbol)
            return None
        adjusted = replace(idea, limit_price=marketable_limit(Side.SELL, quote.price, band))
    else:
        return None
    if adjusted != idea:
        svc.store.save_idea(adjusted)                     # INSERT OR REPLACE: same id, new values
    return adjusted


def entries_blocked(svc: Any) -> bool:
    return bool(svc.store.get_state(PAUSED_KEY)) or bool(svc.store.get_state(MANUAL_STOP_KEY))


async def act_on_ideas(svc: Any, ideas: list[TradeIdea], now: datetime) -> None:
    """Size each idea per book on a fresh quote, then submit it (autopilot ON) or send one approval
    card covering every book (autopilot OFF). Skipped entirely while paused or manually stopped.
    Limits are evaluated first, so a book that just hit a limit takes no entry."""
    async with job_lock(svc, "ideas"):
        await _act_on_ideas(svc, ideas, now)


async def _act_on_ideas(svc: Any, ideas: list[TradeIdea], now: datetime) -> None:
    if entries_blocked(svc):
        log.info("entries are paused or stopped; %d idea(s) not acted on", len(ideas))
        return
    store = svc.store
    held_quotes = await fetch_quotes(svc, held_symbols(svc))
    evals = await check_limits(svc, now, held_quotes)
    autopilot = get_autopilot(store)                      # read after check_limits: a stop turns it off
    acted = list(store.get_state(ACTED_KEY) or [])
    acted_pairs = acted_symbol_sides(store, acted)
    for idea in ideas:
        if idea.id in acted:
            continue
        if (idea.symbol, str(idea.side)) in acted_pairs:
            log.info("%s %s was already acted on today; idea %s skipped", idea.side, idea.symbol, idea.id)
            continue
        # Not marked acted until a card or an order goes out: a missing or stale quote (the delayed
        # feed before about 09:40) leaves it for the next pass of exits_job.
        quote = (await fetch_quotes(svc, [idea.symbol])).get(idea.symbol)
        if quote is None or is_stale(quote, now, svc.settings.risk):
            log.info("%s %s: no fresh quote yet; will try again", idea.side, idea.symbol)
            continue
        idea = reanchor(svc, idea, quote)
        if idea is None:
            continue
        sizes = _sizes(svc, idea, evals, {**held_quotes, idea.symbol: quote}, now)
        if not sizes:
            log.info("idea %s %s: no book can take it (unaffordable, limits or nothing held)",
                     idea.side, idea.symbol)
            continue
        acted.append(idea.id)
        store.set_state(ACTED_KEY, acted)                 # marked before the await: never acted on twice
        acted_pairs.add((idea.symbol, str(idea.side)))
        if autopilot:
            await _submit(svc, idea, sizes, quote, now)
        else:
            await _send_card(svc, idea, sizes, quote, now)


def acted_symbol_sides(store: Any, acted: list[str]) -> set[tuple[str, str]]:
    """(symbol, side) of every idea acted on today (``acted_ideas`` is reset by premarket)."""
    pairs: set[tuple[str, str]] = set()
    for idea_id in acted:
        idea = store.get_idea(idea_id)
        if idea is not None:
            pairs.add((idea.symbol, str(idea.side)))
    return pairs


def unacted_ideas(svc: Any, ideas: list[TradeIdea]) -> list[TradeIdea]:
    acted = set(svc.store.get_state(ACTED_KEY) or [])
    return [i for i in ideas if i.id not in acted]


async def _submit(svc: Any, idea: TradeIdea, sizes: dict[str, int], quote: Quote, now: datetime) -> None:
    for book, qty in sizes.items():
        order = Order(id=uuid.uuid4().hex, book=book, symbol=idea.symbol, side=idea.side, qty=qty,
                      limit_price=idea.limit_price, reason=idea.reason, is_exit=False, idea_id=idea.id,
                      created_at=now)
        try:
            await svc.executor.submit(order, quote, now, idea)
        except Exception as exc:  # noqa: BLE001 - one book failing must not stop the other
            log.error("submitting %s %s for %s failed: %s", idea.side, idea.symbol, book, describe_error(exc))
            await notify(svc, f"{book}: error placing {idea.side} {idea.symbol} ({type(exc).__name__}); "
                              "see the log.")


async def _send_card(svc: Any, idea: TradeIdea, sizes: dict[str, int], quote: Quote, now: datetime) -> None:
    card = svc.approvals.create(idea, sizes, quote.price, now)
    text = svc.approvals.render(card, idea, svc.settings.charges)
    try:
        await svc.notifier.send_card(card, text)
    except Exception as exc:  # noqa: BLE001
        log.error("sending the approval card for %s failed: %s", idea.symbol, describe_error(exc))


def ist_day(now: datetime) -> str:
    return now.astimezone(IST).date().isoformat()
