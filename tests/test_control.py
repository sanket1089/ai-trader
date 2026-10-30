import asyncio
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from exec_fakes import FakeMarketData, FakeNotifier, FakeTime

from aitrader.approvals import ApprovalBook, get_autopilot, set_autopilot
from aitrader.broker import PaperBroker
from aitrader.calendar import at
from aitrader.config import load_settings
from aitrader.control import (approve_recovery, handle_card, handle_command, handle_recovery,
                              pending_recovery)
from aitrader.execution import Executor
from aitrader.models import Order, OrderStatus, Position, Quote, Side, TradeIdea
from aitrader.risk.equity import book_equity
from aitrader.risk.limits import LimitState, LossLimits
from aitrader.risk.manager import RiskManager
from aitrader.store import Store

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
SETTINGS = load_settings(REPO_CONFIG, env={"TELEGRAM_OWNER_CHAT_ID": "42", "TELEGRAM_BOT_TOKEN": "x"})

MON = date(2026, 11, 2)
NOW = at(MON, "10:00")
SYMBOL = "NIFTYBEES"
REAL, LEARN = "realistic", "learning"
UNIVERSE = {"NIFTYBEES", "GOLDBEES"}


class RecordingBroker(PaperBroker):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cancelled: list[str] = []

    def cancel(self, order_id):
        self.cancelled.append(order_id)
        super().cancel(order_id)


class SpyExecutor:
    """Records ``submit`` calls instead of trading; ``raise_for`` makes a book's submit blow up."""

    def __init__(self, raise_for=()):
        self.calls: list[tuple] = []
        self.raise_for = set(raise_for)

    async def submit(self, order, quote, now, idea=None):
        self.calls.append((order, quote, now, idea))
        if order.book in self.raise_for:
            raise RuntimeError("secret-token-123")
        return None


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "t.db"


@pytest.fixture
def svc(db_path):
    store = Store(db_path)
    store.set_cash(REAL, 1000.0)
    store.set_cash(LEARN, 100000.0)
    limits = {b.name: LossLimits(store, b, SETTINGS.holidays) for b in SETTINGS.books}
    risk = RiskManager(store, SETTINGS, limits, UNIVERSE)
    notifier = FakeNotifier()
    broker = RecordingBroker(store, SETTINGS.charges)
    t = FakeTime()
    executor = Executor(store, broker, risk, notifier, dry_run=False, sleep=t.sleep, clock=t.clock)
    market_data = FakeMarketData()
    market_data.set(SYMBOL, 280.0, NOW - timedelta(minutes=1))
    yield SimpleNamespace(store=store, settings=SETTINGS, limits=limits, risk=risk, broker=broker,
                          approvals=ApprovalBook(store, SETTINGS.approval, SETTINGS.risk.entry_end),
                          executor=executor, market_data=market_data, llm=SimpleNamespace(),
                          notifier=notifier)
    store.close()


def run(svc, cmd, *args, now=NOW):
    return handle_command(cmd, list(args), svc, now)


def make_idea(limit=281.0, id="i1"):
    return TradeIdea(id=id, symbol=SYMBOL, side=Side.BUY, limit_price=limit, stop=270.0, target=300.0,
                     expected_move_pct=8.0, confidence=0.7, holding_days=5, reason="trend is up")


def new_card(svc, sizes=None, quote=280.0, now=NOW, idea=None):
    return svc.approvals.create(idea or make_idea(), sizes or {REAL: 3, LEARN: 71}, quote, now)


def hard_stop(svc, book=REAL, equity=400.0):
    assert svc.limits[book].evaluate(equity, NOW).state is LimitState.HARD_STOPPED


def order(id, book=REAL, side=Side.BUY, is_exit=False, status=OrderStatus.PENDING, symbol=SYMBOL, qty=1):
    return Order(id=id, book=book, symbol=symbol, side=side, qty=qty, limit_price=280.0, reason="trend",
                 is_exit=is_exit, idea_id="i1", created_at=NOW, status=status)


def hold(svc, book=REAL, qty=3, avg=280.0):
    svc.store.upsert_position(Position(book=book, symbol=SYMBOL, qty=qty, avg_price=avg, stop=270.0,
                                       target=300.0, opened_at=at(date(2026, 10, 30), "11:00"),
                                       idea_id="i1"))


# ---------------------------------------------------------------- /autopilot


def test_autopilot_on_refused_when_hard_stopped(svc):
    hard_stop(svc)
    reply = run(svc, "autopilot", "on")
    assert "not turned on" in reply and "realistic is hard stopped" in reply
    assert get_autopilot(svc.store) is False


def test_autopilot_on_refused_when_manual_stop(svc):
    svc.store.set_state("manual_stop", True)
    reply = run(svc, "autopilot", "on")
    assert "not turned on" in reply and "manual stop" in reply
    assert get_autopilot(svc.store) is False


def test_autopilot_on_refused_when_paused(svc):
    svc.store.set_state("paused", True)
    reply = run(svc, "autopilot", "on")
    assert "not turned on" in reply and "paused" in reply
    assert get_autopilot(svc.store) is False


def test_autopilot_on_and_off(svc):
    assert "ON" in run(svc, "autopilot", "on")
    assert get_autopilot(svc.store) is True
    assert "OFF" in run(svc, "autopilot", "OFF")
    assert get_autopilot(svc.store) is False


def test_autopilot_off_is_always_allowed(svc):
    set_autopilot(svc.store, True)
    hard_stop(svc)
    svc.store.set_state("manual_stop", True)
    run(svc, "autopilot", "off")
    assert get_autopilot(svc.store) is False


def test_autopilot_allowed_while_only_daily_paused(svc):
    assert svc.limits[REAL].evaluate(850.0, NOW).state is LimitState.DAILY_PAUSED
    assert "ON" in run(svc, "autopilot", "on")
    assert get_autopilot(svc.store) is True


@pytest.mark.parametrize("args", [[], ["maybe"], ["on", "off"]])
def test_autopilot_bad_args_give_usage_and_change_nothing(svc, args):
    reply = run(svc, "autopilot", *args)
    assert "Usage: /autopilot on|off" in reply
    assert get_autopilot(svc.store) is False


# ---------------------------------------------------------------- /stop


def test_stop_cancels_pending_entries_not_exits(svc):
    set_autopilot(svc.store, True)
    for o in (order("e1"), order("e2", book=LEARN), order("s1", side=Side.SELL),
              order("x1", side=Side.SELL, is_exit=True), order("f1", status=OrderStatus.FILLED)):
        svc.store.save_order(o)

    reply = run(svc, "stop")

    status = {i: svc.store.get_order(i).status for i in ("e1", "e2", "s1", "x1", "f1")}
    assert status == {"e1": OrderStatus.CANCELLED, "e2": OrderStatus.CANCELLED,
                      "s1": OrderStatus.CANCELLED, "x1": OrderStatus.PENDING, "f1": OrderStatus.FILLED}
    assert sorted(svc.broker.cancelled) == ["e1", "e2", "s1"]            # the broker was told too
    assert "owner /stop" in svc.store.get_order("e1").reason
    assert get_autopilot(svc.store) is False
    assert svc.store.get_state("manual_stop") is True
    assert svc.store.get_state("tested_stop") is True
    assert "Cancelled 3 pending order(s)" in reply
    assert "1 pending exit order(s) left in place" in reply


def test_stop_with_nothing_pending_still_sets_the_switches(svc):
    set_autopilot(svc.store, True)
    reply = run(svc, "stop")
    assert "No pending entry orders to cancel" in reply
    assert svc.store.get_state("manual_stop") is True
    assert get_autopilot(svc.store) is False


def test_stop_survives_a_cancel_that_fails(svc):
    svc.store.save_order(order("bad"))
    svc.store.save_order(order("good", book=LEARN))

    real = svc.executor

    def cancel_order(order_id, why):
        if order_id == "bad":
            raise RuntimeError("secret-token-123")
        return real.cancel_order(order_id, why)

    svc.executor = SimpleNamespace(cancel_order=cancel_order)
    reply = run(svc, "stop")
    assert svc.store.get_state("manual_stop") is True
    assert svc.store.get_order("good").status is OrderStatus.CANCELLED
    assert svc.store.get_order("bad").status is OrderStatus.PENDING
    assert "Could not cancel" in reply and "secret-token-123" not in reply


def test_stop_without_an_executor_cancel_still_cancels_in_the_store(svc):
    svc.store.save_order(order("e1"))
    svc.executor = SimpleNamespace()
    run(svc, "stop")
    assert svc.store.get_order("e1").status is OrderStatus.CANCELLED
    assert svc.store.get_order("e1").reason.endswith("cancelled: owner /stop")


# ---------------------------------------------------------------- /pause and /resume


def test_pause_blocks_entries_in_risk(svc):
    hold(svc)
    entry = order("new", qty=3)
    quote = Quote(SYMBOL, 280.0, NOW - timedelta(minutes=1))
    assert svc.risk.check(entry, quote, NOW, make_idea(280.0)).approved

    assert "Paused" in run(svc, "pause")

    assert svc.store.get_state("paused") is True
    decision = svc.risk.check(entry, quote, NOW, make_idea(280.0))
    assert not decision.approved and "trading paused by owner" in decision.reasons
    exit_order = order("exit", side=Side.SELL, is_exit=True)
    assert svc.risk.check(exit_order, quote, NOW).approved                 # exits keep working


def test_resume_clears_pause_and_manual_stop(svc):
    svc.store.set_state("paused", True)
    svc.store.set_state("manual_stop", True)
    reply = run(svc, "resume")
    assert svc.store.get_state("paused") is False
    assert svc.store.get_state("manual_stop") is False
    assert "pause and manual stop" in reply
    assert "Autopilot is OFF" in reply
    assert "Nothing was paused or stopped" in run(svc, "resume")


def test_resume_without_recovery_approval_refused(svc):
    hard_stop(svc)
    svc.store.set_state("paused", True)
    svc.store.set_state("manual_stop", True)

    reply = run(svc, "resume")

    assert "still hard stopped" in reply and "recovery plan must be approved first" in reply
    assert svc.limits[REAL].state(NOW) is LimitState.HARD_STOPPED
    assert svc.limits[REAL].baseline() == 1000.0
    assert svc.limits[REAL].stop_id() == 0
    assert svc.store.get_state("paused") is False and svc.store.get_state("manual_stop") is False


@pytest.mark.parametrize("cash, at_risk", [(400.0, "₹400.00"), (700.0, "₹500.00")])
def test_resume_after_recovery_resets_baseline_and_warns(svc, cash, at_risk):
    hard_stop(svc)
    svc.store.set_cash(REAL, cash)
    svc.store.set_state("recovery_approved:realistic:0", True)

    reply = run(svc, "resume")

    assert svc.limits[REAL].baseline() == pytest.approx(cash)
    assert svc.limits[REAL].state(NOW) is LimitState.ACTIVE
    assert svc.limits[REAL].stop_id() == 1
    assert f"New baseline ₹{cash:,.2f}" in reply
    assert f"Up to {at_risk} — possibly everything left — can be lost before the next hard stop" in reply
    assert "₹100.00/day pause still applies" in reply


def test_resume_marks_the_baseline_at_the_fresh_quote(svc):
    hold(svc, qty=2)
    svc.store.set_cash(REAL, 100.0)
    market_price = 200.0
    svc.market_data.set(SYMBOL, market_price, NOW)
    equity = book_equity(svc.store, REAL, {SYMBOL: Quote(SYMBOL, market_price, NOW)}, SETTINGS.charges)
    hard_stop(svc, equity=equity)
    svc.store.set_state("recovery_approved:realistic:0", True)
    svc.market_data.calls.clear()

    run(svc, "resume")

    assert svc.limits[REAL].baseline() == pytest.approx(equity)
    assert svc.market_data.calls == [[SYMBOL]]                               # only that book's holdings


def test_resume_does_not_reset_a_book_that_is_not_hard_stopped(svc):
    svc.store.set_cash(REAL, 400.0)
    svc.store.set_state("recovery_approved:realistic:0", True)
    run(svc, "resume")
    assert svc.limits[REAL].baseline() == 1000.0 and svc.limits[REAL].stop_id() == 0


def test_resume_needs_an_approval_for_the_current_stop_not_an_old_one(svc):
    hard_stop(svc)
    svc.store.set_cash(REAL, 900.0)                       # holdings recovered by the time of /resume
    svc.store.set_state("recovery_approved:realistic:0", True)
    run(svc, "resume")                                    # stop 0 recovered: baseline 900, stop_id is now 1
    assert svc.limits[REAL].stop_id() == 1
    svc.store.set_cash(REAL, 300.0)
    hard_stop(svc, equity=300.0)                          # a second hard stop (loss 600 of the new baseline)

    reply = run(svc, "resume")

    assert "recovery plan must be approved first" in reply
    assert svc.limits[REAL].state(NOW) is LimitState.HARD_STOPPED


def test_resume_says_a_daily_pause_clears_next_day(svc):
    assert svc.limits[REAL].evaluate(850.0, NOW).state is LimitState.DAILY_PAUSED
    assert "daily loss limit" in run(svc, "resume")
    assert svc.limits[REAL].state(NOW) is LimitState.DAILY_PAUSED


# ---------------------------------------------------------------- recovery helpers


def test_pending_recovery_exists_only_while_hard_stopped(svc):
    assert pending_recovery(svc.store, REAL) is None
    hard_stop(svc)
    assert pending_recovery(svc.store, REAL) == {"book": REAL, "stop_id": 0, "approved": False}
    assert pending_recovery(svc.store, LEARN) is None


def test_approve_recovery_sets_the_flag_for_the_current_stop(svc):
    assert approve_recovery(svc.store, REAL) is None                         # not stopped: nothing to approve
    assert svc.store.get_state("recovery_approved:realistic:0") is None
    hard_stop(svc)
    assert approve_recovery(svc.store, REAL) == 0
    assert svc.store.get_state("recovery_approved:realistic:0") is True
    assert pending_recovery(svc.store, REAL)["approved"] is True


def test_handle_recovery_approves_only_the_matching_stop(svc):
    hard_stop(svc)
    stale = handle_recovery(REAL, 7, svc)
    assert "out of date" in stale
    assert svc.store.get_state("recovery_approved:realistic:7") is None
    assert svc.store.get_state("recovery_approved:realistic:0") is None

    ok = handle_recovery(REAL, 0, svc)
    assert ok == "Recovery plan approved — send /resume to restart realistic"
    assert svc.store.get_state("recovery_approved:realistic:0") is True


def test_handle_recovery_for_a_book_that_is_not_stopped_changes_nothing(svc):
    assert "out of date" in handle_recovery(REAL, 0, svc)
    assert svc.store.get_state("recovery_approved:realistic:0") is None


# ---------------------------------------------------------------- /status, /budget, /pnl, /positions, /journal


def test_status_shows_the_whole_picture(svc):
    set_autopilot(svc.store, True)
    svc.store.set_cash(REAL, 950.0)
    svc.store.set_state("last_run:premarket", at(MON, "08:30").isoformat())

    reply = run(svc, "status")

    assert "Autopilot: ON" in reply
    assert "Paused by owner: no" in reply and "Manual stop: no" in reply
    assert "realistic: ACTIVE, headroom today ₹50.00 of ₹100.00, total ₹450.00 of ₹500.00" in reply
    assert "learning: ACTIVE, headroom today ₹10,000.00 of ₹10,000.00, total ₹50,000.00 of ₹50,000.00" in reply
    assert "AI budget: normal" in reply
    assert "premarket 02 Nov 08:30 IST" in reply
    assert "postmarket never" in reply


def test_status_shows_pause_and_stop_flags(svc):
    svc.store.set_state("paused", True)
    svc.store.set_state("manual_stop", True)
    reply = run(svc, "status")
    assert "Paused by owner: YES" in reply and "Manual stop: YES" in reply


def test_status_has_no_headroom_when_hard_stopped_and_never_latches_a_breach(svc):
    svc.store.set_cash(REAL, 400.0)                      # far below both limits, but nothing evaluated yet
    reply = run(svc, "status")
    assert "realistic: ACTIVE, headroom today ₹0.00 of ₹100.00, total ₹0.00 of ₹500.00" in reply
    assert svc.limits[REAL].state(NOW) is LimitState.ACTIVE                  # /status wrote no stop

    hard_stop(svc)
    svc.store.set_cash(REAL, 900.0)                      # equity recovers; the stop stays latched
    assert "realistic: HARD_STOPPED, headroom today ₹0.00 of ₹100.00, total ₹0.00 of ₹500.00" in run(svc, "status")


def test_budget_shows_usd_and_inr(svc):
    svc.store.add_ai_usage("claude-opus-5-5", 1, 1, 0, 0, 10.0, at(date(2026, 11, 1), "10:00"))
    svc.store.add_ai_usage("claude-opus-5-5", 1, 1, 0, 0, 2.5, NOW)
    svc.store.add_ai_usage("claude-opus-5-5", 1, 1, 0, 0, 50.0, at(date(2026, 10, 31), "10:00"))  # last month

    reply = run(svc, "budget")

    assert "November 2026" in reply
    assert "US$12.50" in reply and "₹1,100.00" in reply                     # month: 12.5 x 88
    assert "12.5% of the US$100.00 cap = ₹8,800.00" in reply
    assert "Today: US$2.50 = ₹220.00" in reply
    assert "Level: normal" in reply


@pytest.mark.parametrize("spent, level", [(85.0, "reduced"), (100.0, "exhausted")])
def test_budget_shows_the_level(svc, spent, level):
    svc.store.add_ai_usage("claude-opus-5-5", 1, 1, 0, 0, spent, NOW)
    assert f"Level: {level}" in run(svc, "budget")


def test_pnl_shows_equity_since_baseline_and_ai_spend(svc):
    svc.store.set_cash(REAL, 950.0)
    svc.store.add_ai_usage("claude-opus-5-5", 1, 1, 0, 0, 5.0, NOW)

    reply = run(svc, "pnl")

    assert "realistic: equity ₹950.00; since baseline ₹1,000.00: -₹50.00 (-5.0%)" in reply
    assert "learning: equity ₹100,000.00; since baseline ₹100,000.00: +₹0.00 (+0.0%)" in reply
    assert "AI spend this month: US$5.00 = ₹440.00" in reply


def test_positions_show_holdings_marks_and_pending_orders(svc):
    hold(svc, qty=3)
    svc.market_data.set(SYMBOL, 285.0, NOW)
    svc.store.save_order(order("p1"))

    reply = run(svc, "positions")

    assert "realistic (cash ₹1,000.00):" in reply
    assert "NIFTYBEES 3 @ ₹280.00, last ₹285.00 (+₹15.00 before charges), stop ₹270.00, target ₹300.00" in reply
    assert "pending BUY 1 NIFTYBEES @ limit ₹280.00" in reply
    assert "learning (cash ₹100,000.00):\n  no positions" in reply


def test_positions_without_a_quote_say_so(svc):
    hold(svc)
    svc.market_data.by_symbol.clear()
    assert "no live price" in run(svc, "positions")


def test_journal_lists_the_newest_lessons(svc):
    assert "empty" in run(svc, "journal")
    for i in range(7):
        svc.store.add_lesson(f"lesson {i}", NOW + timedelta(minutes=i))
    reply = run(svc, "journal")
    assert reply.splitlines()[1:] == [f"{n}. lesson {6 - n + 1}" for n in range(1, 6)]
    assert run(svc, "journal", "2").count("\n") == 2


@pytest.mark.parametrize("arg", ["x", "0", "21", "-1"])
def test_journal_bad_count_gives_usage(svc, arg):
    assert run(svc, "journal", arg).startswith("Usage: /journal")


# ---------------------------------------------------------------- robustness of handle_command


def test_unknown_command_gives_help(svc):
    reply = run(svc, "frobnicate")
    assert reply.startswith("Unknown command /frobnicate")
    assert "/status" in reply and "/stop" in reply


def test_help_and_start_give_help(svc):
    assert run(svc, "help").startswith("Commands (owner only):")
    assert run(svc, "start") == run(svc, "help")


def test_command_name_is_normalised(svc):
    assert "Autopilot: OFF" in run(svc, "/STATUS@my_bot")


def test_handle_command_never_raises(svc):
    svc.store.close()
    reply = run(svc, "status")
    assert "/status hit an internal error" in reply
    assert "ProgrammingError" in reply


def test_handle_command_rejects_a_naive_time_without_raising(svc):
    assert "internal error" in run(svc, "status", now=datetime(2026, 11, 2, 10, 0))


# ---------------------------------------------------------------- approval cards


async def test_card_approve_submits_per_book(svc):
    card = new_card(svc)

    reply = await handle_card(card.id, True, svc, NOW)

    assert "Approved BUY NIFTYBEES" in reply
    assert "realistic: filled 3 @ ₹280.84" in reply and "learning: filled 71 @ ₹280.84" in reply
    assert svc.store.positions(REAL)[0].qty == 3
    assert svc.store.positions(LEARN)[0].qty == 71
    assert svc.approvals.get(card.id).status == "approved"
    assert svc.store.get_state("tested_approval") is True
    assert svc.store.get_state("tested_expiry") is None
    assert len(svc.notifier.texts) == 2                                      # the executor announced each fill


async def test_card_approve_builds_orders_from_the_stored_idea(svc):
    svc.executor = spy = SpyExecutor()
    card = new_card(svc)

    await handle_card(card.id, True, svc, NOW)

    assert [c[0].book for c in spy.calls] == [REAL, LEARN]
    for o, quote, now, idea in spy.calls:
        assert (len(o.id), o.symbol, o.side, o.limit_price) == (32, SYMBOL, Side.BUY, 281.0)
        assert (o.reason, o.is_exit, o.idea_id, o.created_at) == ("trend is up", False, "i1", NOW)
        assert o.status is OrderStatus.PENDING
        assert quote == Quote(SYMBOL, 280.0, NOW - timedelta(minutes=1))   # a fresh quote from market data
        assert now == NOW and idea == make_idea()
    assert [c[0].qty for c in spy.calls] == [3, 71]
    assert len({c[0].id for c in spy.calls}) == 2


async def test_card_approve_skips_books_sized_zero(svc):
    svc.executor = spy = SpyExecutor()
    card = new_card(svc, sizes={REAL: 0, LEARN: 71})
    await handle_card(card.id, True, svc, NOW)
    assert [c[0].book for c in spy.calls] == [LEARN]


async def test_card_approve_reports_a_risk_rejection(svc):
    svc.store.set_state("paused", True)
    card = new_card(svc)

    reply = await handle_card(card.id, True, svc, NOW)

    assert "realistic: rejected" in reply and "learning: rejected" in reply
    assert svc.store.positions(REAL) == []
    assert any("trading paused by owner" in t for t in svc.notifier.texts)   # the executor sent the reasons


async def test_card_approve_one_book_failing_does_not_stop_the_other(svc):
    svc.executor = spy = SpyExecutor(raise_for={REAL})
    card = new_card(svc)

    reply = await handle_card(card.id, True, svc, NOW)

    assert [c[0].book for c in spy.calls] == [REAL, LEARN]
    assert "realistic: error placing the order (RuntimeError)" in reply
    assert "secret-token-123" not in reply


async def test_card_reject_places_nothing_and_needs_no_quote(svc):
    svc.executor = spy = SpyExecutor()
    card = new_card(svc)
    svc.market_data.calls.clear()

    reply = await handle_card(card.id, False, svc, NOW)

    assert reply == "Rejected BUY NIFTYBEES. Nothing was ordered."
    assert svc.approvals.get(card.id).status == "rejected"
    assert spy.calls == [] and svc.market_data.calls == []
    assert svc.store.get_state("tested_approval") is None


async def test_card_expired_is_not_placed_and_sets_the_expiry_flag(svc):
    svc.executor = spy = SpyExecutor()
    card = new_card(svc)

    reply = await handle_card(card.id, True, svc, NOW + timedelta(minutes=11))

    assert "expired" in reply
    assert spy.calls == []
    assert svc.store.get_state("tested_expiry") is True
    assert svc.store.get_state("tested_approval") is None


async def test_a_tap_on_a_card_closed_by_a_reissue_is_not_an_expiry_test(svc):
    svc.executor = spy = SpyExecutor()
    card = new_card(svc)
    svc.market_data.set(SYMBOL, 285.0, NOW)
    await handle_card(card.id, True, svc, NOW)                              # price moved: closed, re-issued

    reply = await handle_card(card.id, True, svc, NOW + timedelta(minutes=1))   # a late second tap

    assert "expired" in reply and spy.calls == []
    assert svc.store.get_state("tested_expiry") is None                     # the card did not run out of time


async def test_card_expired_without_a_quote_still_reports_expiry(svc):
    card = new_card(svc)
    svc.market_data.by_symbol.clear()
    reply = await handle_card(card.id, True, svc, NOW + timedelta(minutes=11))
    assert "expired" in reply
    assert svc.store.get_state("tested_expiry") is True


async def test_card_reissue_sends_new_card(svc):
    svc.executor = spy = SpyExecutor()
    card = new_card(svc)                                                     # issued at 280
    svc.market_data.set(SYMBOL, 285.0, NOW)                                  # +1.8%

    reply = await handle_card(card.id, True, svc, NOW)

    assert spy.calls == []
    assert svc.approvals.get(card.id).status == "expired"
    [(new, text)] = svc.notifier.cards
    assert new.id != card.id and new.status == "open"
    assert new.sizes == card.sizes and new.quote_at_issue == 285.0
    assert text == svc.approvals.render(new, make_idea(), SETTINGS.charges)
    assert svc.approvals.get(new.id) == new
    assert "moved more than 0.5%" in reply and "₹285.00" in reply and "fresh card" in reply
    assert svc.store.get_state("tested_approval") is None


async def test_card_reissue_reports_a_card_that_could_not_be_sent(svc):
    card = new_card(svc)
    svc.market_data.set(SYMBOL, 285.0, NOW)

    async def broken(card, text):
        raise RuntimeError("secret-token-123")

    svc.notifier = SimpleNamespace(send_card=broken)
    reply = await handle_card(card.id, True, svc, NOW)
    assert "could not be sent (RuntimeError)" in reply and "secret-token-123" not in reply


async def test_card_without_a_current_price_stays_open(svc):
    svc.executor = spy = SpyExecutor()
    card = new_card(svc)
    svc.market_data.by_symbol.clear()

    reply = await handle_card(card.id, True, svc, NOW)

    assert "No current price" in reply and "stays open" in reply
    assert svc.approvals.get(card.id).status == "open"
    assert spy.calls == []


async def test_card_with_an_unusable_price_is_treated_as_no_price(svc):
    card = new_card(svc)
    svc.market_data.set(SYMBOL, float("nan"), NOW)
    assert "No current price" in await handle_card(card.id, True, svc, NOW)
    assert svc.approvals.get(card.id).status == "open"


async def test_card_double_tap_places_the_orders_once(svc):
    svc.executor = spy = SpyExecutor()
    card = new_card(svc)

    first, second = await asyncio.gather(handle_card(card.id, True, svc, NOW),
                                         handle_card(card.id, True, svc, NOW))

    assert len(spy.calls) == 2                                               # one per book, not four
    assert sorted([first, second], key=len)[0] == "That card was already handled."


async def test_card_unknown_id(svc):
    assert "unknown" in await handle_card("nope", True, svc, NOW)


async def test_card_refuses_to_approve_without_its_idea(svc, db_path):
    svc.executor = spy = SpyExecutor()
    card = new_card(svc)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM ideas WHERE id = 'i1'")

    reply = await handle_card(card.id, True, svc, NOW)

    assert "idea behind this card is missing" in reply
    assert spy.calls == [] and svc.approvals.get(card.id).status == "open"
