"""Owner commands and approval taps as plain functions (the Telegram adapter only carries text).

``handle_command`` answers a slash command with the reply text and never raises: an unknown command
gets the help text, bad arguments get a usage line, and an unexpected error becomes a short message
(only the error's type is logged and shown, never its text). ``handle_card`` turns a tap on an
approval card into orders. ``handle_recovery`` and its two helpers cover the recovery button.

``svc`` is any object with ``store, settings, limits (dict[book, LossLimits]), approvals, executor,
market_data`` and, for a re-issued card, ``notifier`` (the real ``Services`` has all of them).

State keys written here (all via ``store.set_state``):

* ``paused`` and ``manual_stop`` (bool): block new entries; ``RiskManager`` reads them.
* ``recovery_approved:<book>:<stop_id>`` (bool): the owner approved the recovery plan of that stop.
* go-live flags ``tested_approval`` (a card first resolved APPROVED), ``tested_stop`` (first /stop)
  and ``tested_expiry`` (first time a tap met an EXPIRED card).

State keys read here: ``last_run:<job>`` (an aware ISO datetime, written by the jobs) for /status.

``/stop`` and every other command are synchronous and have no ``await``, so on the event loop they
cannot interleave with an order being placed.
"""

import asyncio
import logging
import math
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aitrader.agents.llm import BudgetLevel, budget_level
from aitrader.approvals import Resolution, get_autopilot, set_autopilot
from aitrader.calendar import IST
from aitrader.execution import describe_error, marketable_limit
from aitrader.models import Order, OrderStatus, Quote
from aitrader.risk.equity import book_equity
from aitrader.risk.limits import LimitState
from aitrader.risk.manager import MANUAL_STOP_KEY, PAUSED_KEY
from aitrader.store import Store

log = logging.getLogger(__name__)

TESTED_APPROVAL_KEY = "tested_approval"
TESTED_STOP_KEY = "tested_stop"
TESTED_EXPIRY_KEY = "tested_expiry"

# Jobs whose last run /status shows (the jobs record ``last_run:<job>``).
JOBS = ("premarket", "open_session", "intraday", "exits", "postmarket", "weekly")

_LEVEL_TEXT = {
    BudgetLevel.NORMAL: "normal",
    BudgetLevel.REDUCED: "reduced (effort forced to low, intraday AI re-runs at most every 2 hours)",
    BudgetLevel.EXHAUSTED: "exhausted (no AI calls; exits keep working)",
}

HELP = "\n".join([
    "Commands (owner only):",
    "/status - autopilot, limits and headroom, AI budget, last job runs",
    "/autopilot on|off - place ideas automatically, or ask first",
    "/positions - holdings and pending orders per book",
    "/pnl - equity and profit/loss per book",
    "/journal [n] - the latest lessons",
    "/pause - block new entries (exits still work)",
    "/resume - lift pause and manual stop; restart a hard-stopped book once its recovery plan is approved",
    "/stop - kill switch: cancel pending entries, block new ones, autopilot off",
    "/budget - AI spend this month and today",
])


# ---------------------------------------------------------------- small helpers

def _inr(x: float, signed: bool = False) -> str:
    x = round(x, 2) + 0.0                       # + 0.0 turns -0.0 into 0.0: never print "-₹0.00"
    if signed:
        return f"{'+' if x >= 0 else '-'}₹{abs(x):,.2f}"
    return f"₹{x:,.2f}"


def _usd(x: float) -> str:
    return f"US${x:,.2f}"


def _usable(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _aware(now: datetime) -> datetime:
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be a timezone-aware datetime")
    return now.astimezone(IST)


def _recovery_key(book: str, stop_id: int) -> str:
    return f"recovery_approved:{book}:{stop_id}"


def _describe_order(o: Order) -> str:
    return f"{o.book} {o.side} {o.qty} {o.symbol} @ limit {_inr(o.limit_price)}"


def _quotes(svc: Any, books: Iterable[str]) -> dict[str, Quote]:
    """Quotes for everything the given books hold. A feed failure is logged and gives no quotes."""
    symbols = sorted({p.symbol for b in books for p in svc.store.positions(b)})
    if not symbols:
        return {}
    try:
        return dict(svc.market_data.quotes(symbols))
    except Exception as exc:  # noqa: BLE001 - a down feed must not break a status command
        log.warning("quotes unavailable: %s", describe_error(exc))
        return {}


def _unpriced(svc: Any, book: str, quotes: dict[str, Quote]) -> list[str]:
    """Held symbols with no usable quote (``book_equity`` marks those at cost)."""
    return [p.symbol for p in svc.store.positions(book)
            if p.symbol not in quotes or not _usable(quotes[p.symbol].price)]


def _cost_note(symbols: list[str]) -> str:
    return f" (no live price for {', '.join(symbols)}: marked at cost)" if symbols else ""


def _equity(svc: Any, book: str, quotes: dict[str, Quote]) -> float:
    return book_equity(svc.store, book, quotes, svc.settings.charges)


def _books(svc: Any) -> dict:
    return {b.name: b for b in svc.settings.books}


def _last_run_text(value: object) -> str:
    if value is None:
        return "never"
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)[:40]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(IST).strftime("%d %b %H:%M") + " IST"


# ---------------------------------------------------------------- commands

def _status(args: list[str], svc: Any, now: datetime) -> str:
    store = svc.store
    books = _books(svc)
    quotes = _quotes(svc, svc.limits)
    lines = [f"Autopilot: {'ON' if get_autopilot(store) else 'OFF'}",
             f"Paused by owner: {'YES' if store.get_state(PAUSED_KEY) else 'no'}",
             f"Manual stop: {'YES' if store.get_state(MANUAL_STOP_KEY) else 'no'}"]
    for name, limits in svc.limits.items():
        cfg = books.get(name)
        state = limits.state(now)
        if cfg is None:
            lines.append(f"{name}: {state.value}")
            continue
        try:
            equity = _equity(svc, name, quotes)
            daily_loss, total_loss = limits.losses(equity, now)
        except (KeyError, ValueError) as exc:
            lines.append(f"{name}: {state.value}, headroom unavailable ({type(exc).__name__})")
            continue
        # A latched pause or stop leaves no headroom even if equity has since recovered.
        daily = 0.0 if state is not LimitState.ACTIVE else max(0.0, cfg.daily_loss_limit - daily_loss)
        total = 0.0 if state is LimitState.HARD_STOPPED else max(0.0, cfg.total_loss_limit - total_loss)
        lines.append(f"{name}: {state.value}, headroom today {_inr(daily)} of {_inr(cfg.daily_loss_limit)}, "
                     f"total {_inr(total)} of {_inr(cfg.total_loss_limit)}"
                     f"{_cost_note(_unpriced(svc, name, quotes))}")
    ai = svc.settings.ai
    local = now.astimezone(IST)
    spent = store.ai_spend_usd(local.year, local.month)
    lines.append(f"AI budget: {budget_level(store, ai, now).value} "
                 f"({_usd(spent)} of {_usd(ai.monthly_budget_usd)} this month)")
    lines.append("Last runs: " + ", ".join(f"{job} {_last_run_text(store.get_state(f'last_run:{job}'))}"
                                           for job in JOBS))
    return "\n".join(lines)


def _autopilot(args: list[str], svc: Any, now: datetime) -> str:
    store = svc.store
    usage = "Usage: /autopilot on|off"
    if len(args) != 1 or args[0].lower() not in ("on", "off"):
        return f"Autopilot is {'ON' if get_autopilot(store) else 'OFF'}. {usage}"
    if args[0].lower() == "off":
        set_autopilot(store, False)
        return "Autopilot OFF: every idea needs your approval card."
    why = [f"{name} is hard stopped (total loss limit reached)"
           for name, limits in svc.limits.items() if limits.state(now) is LimitState.HARD_STOPPED]
    if store.get_state(MANUAL_STOP_KEY):
        why.append("a manual stop is active (/stop)")
    if store.get_state(PAUSED_KEY):
        why.append("trading is paused (/pause)")
    if why:
        return "Autopilot not turned on: " + "; ".join(why) + ". Send /resume first."
    set_autopilot(store, True)
    return "Autopilot ON: ideas are placed automatically through the risk gate, without approval cards."


def _positions(args: list[str], svc: Any, now: datetime) -> str:
    store = svc.store
    quotes = _quotes(svc, svc.limits)
    lines: list[str] = []
    for name in svc.limits:
        try:
            cash = f"cash {_inr(store.cash(name))}"
        except KeyError:
            cash = "cash not set"
        lines.append(f"{name} ({cash}):")
        positions = store.positions(name)
        if not positions:
            lines.append("  no positions")
        for p in positions:
            q = quotes.get(p.symbol)
            if q is not None and _usable(q.price):
                gain = (q.price - p.avg_price) * p.qty
                last = f"last {_inr(q.price)} ({_inr(gain, True)} before charges)"
            else:
                last = "no live price"
            lines.append(f"  {p.symbol} {p.qty} @ {_inr(p.avg_price)}, {last}, "
                         f"stop {_inr(p.stop)}, target {_inr(p.target)}")
        for o in store.pending_orders(name):
            lines.append(f"  pending {'exit ' if o.is_exit else ''}{o.side} {o.qty} {o.symbol} "
                         f"@ limit {_inr(o.limit_price)}")
    return "\n".join(lines)


def _pnl(args: list[str], svc: Any, now: datetime) -> str:
    store = svc.store
    quotes = _quotes(svc, svc.limits)
    lines: list[str] = []
    for name, limits in svc.limits.items():
        try:
            equity = _equity(svc, name, quotes)
            daily_loss, total_loss = limits.losses(equity, now)
        except (KeyError, ValueError) as exc:
            lines.append(f"{name}: unavailable ({type(exc).__name__})")
            continue
        base = limits.baseline()
        since = -total_loss
        pct = round(since / base * 100, 1) + 0.0 if base else 0.0
        lines.append(f"{name}: equity {_inr(equity)}; since baseline {_inr(base)}: "
                     f"{_inr(since, True)} ({pct:+.1f}%); since the last close: {_inr(-daily_loss, True)}"
                     f"{_cost_note(_unpriced(svc, name, quotes))}")
    ai = svc.settings.ai
    local = now.astimezone(IST)
    spent = store.ai_spend_usd(local.year, local.month)
    lines.append(f"AI spend this month: {_usd(spent)} = {_inr(spent * ai.usd_inr)} "
                 "(a learning cost, counted apart from trading P&L)")
    return "\n".join(lines)


def _journal(args: list[str], svc: Any, now: datetime) -> str:
    count = 5
    if args:
        if len(args) != 1 or not args[0].isdigit() or not 1 <= int(args[0]) <= 20:
            return "Usage: /journal [count 1-20]"
        count = int(args[0])
    lessons = svc.store.recent_lessons(count)
    if not lessons:
        return "The journal is empty: no lessons yet."
    return "Latest lessons, newest first:\n" + "\n".join(
        f"{i}. {text if len(text) <= 500 else text[:497] + '...'}" for i, text in enumerate(lessons, 1))


def _pause(args: list[str], svc: Any, now: datetime) -> str:
    svc.store.set_state(PAUSED_KEY, True)
    return ("Paused: no new entries in any book until /resume. Exits and sells still work. "
            "(/stop also cancels pending orders.)")


def _stop(args: list[str], svc: Any, now: datetime) -> str:
    store = svc.store
    # The switches first: whatever happens while cancelling, nothing new can start.
    store.set_state(MANUAL_STOP_KEY, True)
    set_autopilot(store, False)
    store.set_state(TESTED_STOP_KEY, True)
    if not callable(getattr(svc.executor, "cancel_order", None)):
        # No store-only fallback: a kill switch must never pretend to cancel a live order. The
        # switches above are already set, so nothing new can start; handle_command tells the owner.
        raise ExecutorCannotCancel("the executor has no cancel_order; pending orders were not cancelled")

    cancelled: list[Order] = []
    failed: list[Order] = []
    exits = 0
    for order in store.pending_orders():
        if order.is_exit:
            exits += 1                                  # a stop/target exit keeps protecting the holding
            continue
        try:
            if _cancel_pending(svc, order):
                cancelled.append(order)
        except Exception as exc:  # noqa: BLE001 - one bad order must not stop the others being cancelled
            log.error("cancelling order %s failed: %s", order.id, describe_error(exc))
            failed.append(order)

    lines = ["STOP: manual stop is ON and autopilot is OFF. No new entries in any book."]
    if cancelled:
        lines.append(f"Cancelled {len(cancelled)} pending order(s):")
        lines += [f"  {_describe_order(o)}" for o in cancelled]
    else:
        lines.append("No pending entry orders to cancel.")
    if failed:
        lines.append("Could not cancel (see the log): " + "; ".join(_describe_order(o) for o in failed))
    if exits:
        lines.append(f"{exits} pending exit order(s) left in place: stop and target exits keep working.")
    lines.append("Send /resume to lift the stop.")
    return "\n".join(lines)


class ExecutorCannotCancel(RuntimeError):
    """The executor has no ``cancel_order``. /stop fails loudly rather than mark a live order
    cancelled in the store while the broker still holds it."""


def _cancel_pending(svc: Any, order: Order) -> bool:
    """Cancel a pending order through the executor (broker and store)."""
    return bool(svc.executor.cancel_order(order.id, "cancelled: owner /stop"))


def _resume(args: list[str], svc: Any, now: datetime) -> str:
    store = svc.store
    was = [label for key, label in ((PAUSED_KEY, "pause"), (MANUAL_STOP_KEY, "manual stop"))
           if store.get_state(key)]
    store.set_state(PAUSED_KEY, False)
    store.set_state(MANUAL_STOP_KEY, False)
    lines = [f"Resumed: cleared the {' and '.join(was)}." if was else "Nothing was paused or stopped."]

    books = _books(svc)
    for name, limits in svc.limits.items():
        state = limits.state(now)
        if state is LimitState.DAILY_PAUSED:
            lines.append(f"{name}: paused for the day by its daily loss limit; that clears next trading day.")
        elif state is LimitState.HARD_STOPPED:
            lines.append(_resume_hard_stopped(svc, name, limits, books[name]))
    if not get_autopilot(store):
        lines.append("Autopilot is OFF: send /autopilot on when you want it.")
    return "\n".join(lines)


def _resume_hard_stopped(svc: Any, name: str, limits: Any, cfg: Any) -> str:
    """Lift a hard stop only if its recovery plan was approved; the new baseline is the current equity."""
    stop_id = limits.stop_id()
    if svc.store.get_state(_recovery_key(name, stop_id)) is not True:
        return (f"{name}: still hard stopped. A recovery plan must be approved first "
                "(tap the ✅ button on its recovery plan message), then send /resume again.")
    quotes = _quotes(svc, [name])
    try:
        equity = _equity(svc, name, quotes)
        limits.reset_baseline(equity)
    except (KeyError, ValueError) as exc:
        log.error("baseline reset for %s failed: %s", name, describe_error(exc))
        return f"{name}: could not reset the baseline ({type(exc).__name__}); still hard stopped."
    return (f"{name}: hard stop lifted. New baseline {_inr(equity)} (current equity"
            f"{_cost_note(_unpriced(svc, name, quotes))}). "
            f"Up to {_inr(min(cfg.total_loss_limit, equity))} — possibly everything left — can be lost "
            f"before the next hard stop; the {_inr(cfg.daily_loss_limit)}/day pause still applies.")


def _budget(args: list[str], svc: Any, now: datetime) -> str:
    ai = svc.settings.ai
    local = now.astimezone(IST)
    month = svc.store.ai_spend_usd(local.year, local.month)
    today = svc.store.ai_spend_usd_on(local.date())
    cap = ai.monthly_budget_usd
    pct = month / cap * 100 if cap > 0 else 0.0
    return "\n".join([
        f"AI budget, {local.strftime('%B %Y')}:",
        f"Month so far: {_usd(month)} = {_inr(month * ai.usd_inr)} "
        f"({pct:.1f}% of the {_usd(cap)} cap = {_inr(cap * ai.usd_inr)}, at ₹{ai.usd_inr:g}/US$)",
        f"Today: {_usd(today)} = {_inr(today * ai.usd_inr)}",
        f"Level: {_LEVEL_TEXT[budget_level(svc.store, ai, now)]}",
    ])


def _help(args: list[str], svc: Any, now: datetime) -> str:
    return HELP


_COMMANDS: dict[str, Callable[[list[str], Any, datetime], str]] = {
    "status": _status, "autopilot": _autopilot, "positions": _positions, "pnl": _pnl,
    "journal": _journal, "pause": _pause, "resume": _resume, "stop": _stop, "budget": _budget,
    "help": _help, "start": _help,
}
COMMAND_NAMES = tuple(_COMMANDS)


def handle_command(cmd: str, args: list[str], svc: Any, now: datetime) -> str:
    """Reply text for ``/cmd args``. Never raises."""
    name = str(cmd).strip().lstrip("/").split("@", 1)[0].lower()
    handler = _COMMANDS.get(name)
    if handler is None:
        return f"Unknown command /{name[:30]}.\n{HELP}" if name else HELP
    try:
        return handler([str(a) for a in (args or [])], svc, _aware(now))
    except Exception as exc:  # noqa: BLE001 - the owner always gets an answer; only the type is shown
        log.error("command /%s failed: %s", name, describe_error(exc))
        return (f"/{name} hit an internal error ({type(exc).__name__}); see the log. "
                "Send /status to see where things stand.")


# ---------------------------------------------------------------- approval cards

async def _fresh_quote(svc: Any, symbol: str) -> Quote | None:
    """The latest quote for ``symbol`` (the feed is blocking, so it runs in a thread), or ``None``
    when there is none or its price is unusable."""
    try:
        quotes = await asyncio.to_thread(svc.market_data.quotes, [symbol])
    except Exception as exc:  # noqa: BLE001
        log.warning("no fresh quote for %s: %s", symbol, describe_error(exc))
        return None
    quote = quotes.get(symbol)
    return quote if quote is not None and _usable(quote.price) else None


@dataclass(frozen=True)
class CardResult:
    """The reply to a card tap, and whether the card is still open (so its buttons must stay)."""

    text: str
    still_open: bool = False


async def handle_card(card_id: str, approve: bool, svc: Any, now: datetime) -> str:
    """Apply the owner's tap on a card and return the reply text (see ``handle_card_tap``)."""
    return (await handle_card_tap(card_id, approve, svc, now)).text


async def handle_card_tap(card_id: str, approve: bool, svc: Any, now: datetime) -> CardResult:
    """Apply the owner's tap on a card.

    The idea is always loaded from the store by the card's ``idea_id``; nothing else is trusted.
    APPROVED places one order per book with a size above 0 through ``svc.executor.submit``;
    REISSUE closes the card and sends a fresh one at the current price. ``still_open`` is True when
    the tap left the card unresolved (no current price, or its idea is missing): the adapter then
    keeps the card's buttons so the owner can tap again.
    """
    card = svc.approvals.get(card_id)
    if card is None:
        return CardResult("That approval card is unknown.")
    idea = svc.store.get_idea(card.idea_id)

    quote: Quote | None = None
    if approve:
        if idea is None:
            log.error("card %s: its idea %s is missing; not approving", card_id, card.idea_id)
            return CardResult("Cannot approve: the trade idea behind this card is missing. "
                              "Nothing was ordered.", still_open=card.status == "open")
        quote = await _fresh_quote(svc, idea.symbol)
        if quote is None and card.status == "open" and now < card.expires_at:
            return CardResult(f"No current price for {idea.symbol} right now, so the card was not "
                              "resolved. It stays open: tap again in a moment.", still_open=True)
    # A reject needs no price; an approve with no price on a card past its time still resolves (EXPIRED).
    price = quote.price if quote is not None else (math.nan if approve else card.quote_at_issue)
    resolution = svc.approvals.resolve(card_id, approve, price, now)

    what = f"{idea.side} {idea.symbol}" if idea is not None else "the trade"
    if resolution is Resolution.APPROVED:
        svc.store.set_state(TESTED_APPROVAL_KEY, True)
        return CardResult(await _place_orders(svc, card, idea, quote, now))
    if resolution is Resolution.REJECTED:
        return CardResult(f"Rejected {what}. Nothing was ordered.")
    if resolution is Resolution.EXPIRED:
        if now >= card.expires_at:
            # Only a card that ran out of time counts as "expiry tested": a card closed early by a
            # re-issue is also EXPIRED, and a late second tap on it must not tick the go-live box.
            svc.store.set_state(TESTED_EXPIRY_KEY, True)
        return CardResult(f"This card for {what} has expired and was not placed. Wait for the next one.")
    if resolution is Resolution.ALREADY_RESOLVED:
        return CardResult("That card was already handled.")
    if resolution is Resolution.REISSUE:
        return CardResult(await _reissue(svc, card, idea, quote, now))
    return CardResult("That approval card is unknown.")


async def _place_orders(svc: Any, card: Any, idea: Any, quote: Quote | None, now: datetime) -> str:
    lines = [f"Approved {idea.side} {idea.symbol}."]
    # The limit is re-set at the price of this moment: the idea's was set when the card was issued, the
    # price has moved since (up to the re-issue tolerance), and a stale limit is either refused by the
    # risk gate's band or too low to fill once the paper broker adds its slippage.
    limit = (marketable_limit(idea.side, quote.price, svc.settings.risk.entry_band_pct)
             if quote is not None else idea.limit_price)
    for book, qty in card.sizes.items():
        if qty <= 0:
            continue
        order = Order(id=uuid.uuid4().hex, book=book, symbol=idea.symbol, side=idea.side, qty=qty,
                      limit_price=limit, reason=idea.reason, is_exit=False, idea_id=idea.id,
                      created_at=now)
        try:
            fill = await svc.executor.submit(order, quote, now, idea)
        except Exception as exc:  # noqa: BLE001 - one book failing must not stop the other
            log.error("placing the %s order for %s failed: %s", book, idea.symbol, describe_error(exc))
            lines.append(f"{book}: error placing the order ({type(exc).__name__}); see the log")
            continue
        lines.append(f"{book}: {_order_outcome(svc.store, order, fill)}")
    return "\n".join(lines)


def _order_outcome(store: Store, order: Order, fill: Any) -> str:
    if fill is not None:
        return f"filled {fill.qty} @ {_inr(fill.price)}, charges {_inr(fill.charges)}"
    stored = store.get_order(order.id)
    status = stored.status if stored is not None else order.status
    if status is OrderStatus.REJECTED:
        return "rejected (the reasons were sent as an alert)"
    if status is OrderStatus.CANCELLED:
        return "not placed (dry-run)"
    return f"{order.qty} placed at limit {_inr(order.limit_price)}, waiting to fill"


async def _reissue(svc: Any, card: Any, idea: Any, quote: Quote | None, now: datetime) -> str:
    if idea is None or quote is None:
        return "The price could not be confirmed and the card was closed. Wait for the next one."
    new_card = svc.approvals.create(idea, card.sizes, quote.price, now)
    text = svc.approvals.render(new_card, idea, svc.settings.charges)
    tolerance = svc.settings.approval.reprice_tolerance_pct
    head = (f"The price of {idea.symbol} moved more than {tolerance:g}% since the card was issued "
            f"(now {_inr(quote.price)}), so this card was closed.")
    try:
        await svc.notifier.send_card(new_card, text)
    except Exception as exc:  # noqa: BLE001
        log.error("sending the fresh card failed: %s", describe_error(exc))
        return f"{head} A fresh card was made but could not be sent ({type(exc).__name__}); see the log."
    return f"{head} A fresh card is on its way: approve it if you still want the trade."


# ---------------------------------------------------------------- recovery

def pending_recovery(store: Store, book: str) -> dict | None:
    """``{"book", "stop_id", "approved"}`` while ``book`` is hard stopped, else ``None``.

    Reads the keys ``LossLimits`` persists (``hard_stop:<book>`` and ``stop_id:<book>``).
    """
    if store.get_state(f"hard_stop:{book}") is not True:
        return None
    stop_id = int(store.get_state(f"stop_id:{book}", 0))
    approved = store.get_state(_recovery_key(book, stop_id)) is True
    return {"book": book, "stop_id": stop_id, "approved": approved}


def approve_recovery(store: Store, book: str) -> int | None:
    """Approve the recovery plan of ``book``'s current hard stop. Returns the stop id approved, or
    ``None`` (and changes nothing) when the book is not hard stopped."""
    pending = pending_recovery(store, book)
    if pending is None:
        return None
    store.set_state(_recovery_key(book, pending["stop_id"]), True)
    return pending["stop_id"]


def handle_recovery(book: str, stop_id: int, svc: Any) -> str:
    """The ``recovery:<book>:<stop_id>`` button: approves only the stop the button was sent for."""
    pending = pending_recovery(svc.store, book)
    if pending is None or pending["stop_id"] != stop_id:
        return f"This recovery plan is out of date (no matching hard stop for {book}). Nothing changed."
    approve_recovery(svc.store, book)
    return f"Recovery plan approved — send /resume to restart {book}"
