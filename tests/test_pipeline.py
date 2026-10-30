"""Pipeline jobs end to end with fake market data, a fake LLM and a recording notifier (offline)."""

import asyncio
import time
from datetime import date, timedelta

import pytest
from pipeline_fakes import LEARN, REAL, FakeLLM, FakeMarketData, RecordingNotifier, make_svc

from aitrader import pipeline
from aitrader.agents.schemas import (
    IdeaOut,
    NewsDigest,
    PMDecision,
    RecoveryPlan,
    ReviewOutput,
    SymbolNews,
    TechnicalBatch,
    TechnicalView,
)
from aitrader.approvals import get_autopilot, set_autopilot
from aitrader.calendar import at
from aitrader.charges import compute_charges
from aitrader.models import Fill, Order, Position, Side, TradeIdea
from aitrader.risk.limits import LimitState
from aitrader.risk.manager import MANUAL_STOP_KEY, PAUSED_KEY

MON = date(2026, 11, 2)
PRICES = {"NIFTYBEES": 280.0, "GOLDBEES": 100.0, "RELIANCE": 1300.0, "ITC": 400.0}


def md_at(now, prices=PRICES, history=True):
    return FakeMarketData(prices, now - timedelta(minutes=16), history=history)


def nifty_idea(id="idea1", symbol="NIFTYBEES", price=280.0):
    return TradeIdea(id=id, symbol=symbol, side=Side.BUY, limit_price=price, stop=price * 0.95,
                     target=price * 1.08, expected_move_pct=8.0, confidence=0.7, holding_days=5,
                     reason="breakout")


def pm_decision(symbol="NIFTYBEES", price=280.0):
    return PMDecision(market_view="calm", ideas=[IdeaOut(
        symbol=symbol, side="BUY", limit_price=price, stop=price * 0.95, target=price * 1.08,
        expected_move_pct=8.0, confidence=0.7, holding_days=5, reason="breakout")])


def full_llm():
    return FakeLLM({
        "NewsDigest": NewsDigest(market_mood="neutral", market_summary="quiet", symbols=[
            SymbolNews(symbol="NIFTYBEES", sentiment=0.2, materiality="low", summary="flat")]),
        "TechnicalBatch": TechnicalBatch(views=[TechnicalView(
            symbol="NIFTYBEES", trend="up", support=270, resistance=300, signal="buy", confidence=0.6,
            notes="ok")]),
        "PMDecision": pm_decision(),
        "RecoveryPlan": RecoveryPlan(what_went_wrong="chased momentum", changes=["smaller size"],
                                     mode="exit_only", next_trades=[]),
        "ReviewOutput": ReviewOutput(lessons=["Wait for the pullback"], trade_reviews=[]),
    })


def news_feed(title="Nifty rallies as RBI holds rates"):
    published = time.struct_time((2026, 11, 2, 2, 0, 0, 0, 0, 0))     # 07:30 IST
    return {"feed": {"title": "F"}, "bozo": 0,
            "entries": [{"title": title, "link": "https://n/1", "summary": "", "published_parsed": published}]}


def stage_ideas(svc, ideas, day=MON):
    for idea in ideas:
        svc.store.save_idea(idea)
    svc.store.set_state("todays_ideas", [i.id for i in ideas])
    svc.store.set_state("todays_ideas_day", day.isoformat())


# ---------------------------------------------------------------- premarket

async def test_premarket_data_down_alerts_and_skips_ai(tmp_path):
    now = at(MON, "08:30")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now, prices={}), llm, now=now)
    await pipeline.premarket(svc, now)
    assert any("data feed down" in t.lower() for t in svc.notifier.texts)
    assert llm.calls == []
    assert svc.store.get_state("todays_ideas") is None


async def test_premarket_quote_missing_but_history_ok_runs_ai(tmp_path):
    now = at(MON, "08:30")
    llm = full_llm()
    md = md_at(now)
    md.quotes = lambda symbols: {}                    # no intraday quote before the open
    svc = make_svc(tmp_path, md, llm, now=now)
    await pipeline.premarket(svc, now)
    assert not any("data feed down" in t.lower() for t in svc.notifier.texts)
    assert ("technical", "TechnicalBatch") in llm.calls
    assert len(svc.store.get_state("todays_ideas")) == 1      # PM priced off the last daily close


async def test_premarket_both_quote_and_history_missing_alerts(tmp_path):
    now = at(MON, "08:30")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now, history=False), llm, now=now)
    svc.market_data.quotes = lambda symbols: {}
    await pipeline.premarket(svc, now)
    assert any("data feed down" in t.lower() for t in svc.notifier.texts)
    assert llm.calls == []


async def test_premarket_stores_todays_ideas(tmp_path):
    now = at(MON, "08:30")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now), llm, now=now, feeds=["https://feed"])
    svc.news_parse = lambda url: news_feed()
    await pipeline.premarket(svc, now)

    ids = svc.store.get_state("todays_ideas")
    assert len(ids) == 1
    idea = svc.store.get_idea(ids[0])
    assert idea is not None and idea.symbol == "NIFTYBEES"
    assert svc.store.get_state("todays_ideas_day") == MON.isoformat()
    assert [c[1] for c in llm.calls] == ["NewsDigest", "TechnicalBatch", "PMDecision"]
    assert svc.store.get_state("news_since") == now.isoformat()
    # held symbols are always in the shortlist, and the realistic book gets affordable candidates
    shortlist = svc.store.get_state("shortlist")
    assert "NIFTYBEES" in shortlist


async def test_premarket_shortlist_keeps_held_and_caps_realistic_candidates(tmp_path):
    now = at(MON, "08:30")
    svc = make_svc(tmp_path, md_at(now), full_llm(), now=now)
    svc.store.upsert_position(Position(LEARN, "RELIANCE", 1, 1300.0, 1200.0, 1400.0, now))
    await pipeline.premarket(svc, now)
    shortlist = svc.store.get_state("shortlist")
    assert shortlist[0] == "RELIANCE"
    assert set(shortlist) >= {"RELIANCE", "NIFTYBEES", "GOLDBEES", "ITC"}


# ---------------------------------------------------------------- open session

async def test_open_session_autopilot_off_sends_cards(tmp_path):
    now = at(MON, "09:20")
    svc = make_svc(tmp_path, md_at(now), now=now)
    stage_ideas(svc, [nifty_idea()])
    await pipeline.open_session(svc, now)
    assert len(svc.notifier.cards) == 1
    card, text = svc.notifier.cards[0]
    assert set(card.sizes) == {REAL, LEARN} and all(q > 0 for q in card.sizes.values())
    assert "NIFTYBEES" in text and "charges" in text
    assert svc.executor.submits == []


async def test_open_session_autopilot_on_submits_both_books(tmp_path):
    now = at(MON, "09:20")
    svc = make_svc(tmp_path, md_at(now), now=now)
    set_autopilot(svc.store, True)
    stage_ideas(svc, [nifty_idea()])
    await pipeline.open_session(svc, now)
    books = {o.book: o for o, *_ in svc.executor.submits}
    assert set(books) == {REAL, LEARN}
    assert books[REAL].qty == 3 and books[REAL].side is Side.BUY and books[REAL].idea_id == "idea1"
    assert books[LEARN].qty > books[REAL].qty
    assert svc.notifier.cards == []
    # running it again never places the same idea twice
    await pipeline.open_session(svc, now)
    assert len(svc.executor.submits) == 2


@pytest.mark.parametrize("key", [PAUSED_KEY, MANUAL_STOP_KEY])
async def test_open_session_skips_when_paused(tmp_path, key):
    now = at(MON, "09:20")
    svc = make_svc(tmp_path, md_at(now), now=now)
    set_autopilot(svc.store, True)
    svc.store.set_state(key, True)
    stage_ideas(svc, [nifty_idea()])
    await pipeline.open_session(svc, now)
    assert svc.executor.submits == [] and svc.notifier.cards == []


async def test_open_session_ignores_ideas_from_an_earlier_day(tmp_path):
    now = at(MON, "09:20")
    svc = make_svc(tmp_path, md_at(now), now=now)
    stage_ideas(svc, [nifty_idea()], day=MON - timedelta(days=3))
    await pipeline.open_session(svc, now)
    assert svc.notifier.cards == []


async def test_open_session_reanchors_buy_limit_to_fresh_quote(tmp_path):
    now = at(MON, "09:20")
    svc = make_svc(tmp_path, md_at(now, prices={**PRICES, "NIFTYBEES": 284.0}), now=now, spy=False)
    idea = TradeIdea(id="i1", symbol="NIFTYBEES", side=Side.BUY, limit_price=280.0, stop=266.0, target=300.0,
                     expected_move_pct=7.0, confidence=0.7, holding_days=5, reason="breakout")
    stage_ideas(svc, [idea])
    await pipeline.open_session(svc, now)

    stored = svc.store.get_idea("i1")
    assert stored.limit_price == 284.0 and stored.stop == 266.0 and stored.target == 300.0
    assert stored.expected_move_pct == pytest.approx((300.0 - 284.0) / 284.0 * 100)
    card, text = svc.notifier.cards[0]
    assert "₹284.00" in text
    quote = svc.market_data.quote("NIFTYBEES")
    order = Order(id="o", book=LEARN, symbol="NIFTYBEES", side=Side.BUY, qty=card.sizes[LEARN],
                  limit_price=stored.limit_price, reason="r", is_exit=False, idea_id="i1", created_at=now)
    decision = svc.risk.check(order, quote, now, stored)
    assert not any("outside" in r for r in decision.reasons), decision.reasons
    assert decision.approved, decision.reasons


async def test_open_session_skips_idea_already_beyond_target(tmp_path):
    now = at(MON, "09:20")
    svc = make_svc(tmp_path, md_at(now, prices={**PRICES, "NIFTYBEES": 305.0}), now=now)
    set_autopilot(svc.store, True)
    stage_ideas(svc, [nifty_idea()])                       # target 302.4
    await pipeline.open_session(svc, now)
    assert svc.executor.submits == [] and svc.notifier.cards == []
    assert svc.store.get_idea("idea1").limit_price == 280.0


async def test_open_session_sell_idea_reanchored_only_when_held(tmp_path):
    now = at(MON, "09:20")
    svc = make_svc(tmp_path, md_at(now, prices={**PRICES, "GOLDBEES": 101.234}), now=now)
    sell = TradeIdea(id="s1", symbol="GOLDBEES", side=Side.SELL, limit_price=100.0, stop=100.0, target=100.0,
                     expected_move_pct=0.0, confidence=0.6, holding_days=1, reason="exit early")
    stage_ideas(svc, [sell])
    await pipeline.open_session(svc, now)
    assert svc.notifier.cards == []                        # not held: skipped

    svc.store.upsert_position(Position(LEARN, "GOLDBEES", 5, 95.0, 90.0, 120.0, now))
    svc.store.set_state("acted_ideas", [])
    await pipeline.open_session(svc, now)
    assert svc.store.get_idea("s1").limit_price == 101.23
    assert svc.notifier.cards[0][0].sizes == {LEARN: 5}


async def test_realistic_book_skipped_when_unaffordable(tmp_path):
    now = at(MON, "09:20")
    svc = make_svc(tmp_path, md_at(now), now=now)
    stage_ideas(svc, [nifty_idea(symbol="RELIANCE", price=1300.0)])
    await pipeline.open_session(svc, now)
    card, _ = svc.notifier.cards[0]
    assert set(card.sizes) == {LEARN}


async def test_open_session_hard_stopped_book_gets_no_entry(tmp_path):
    now = at(MON, "09:20")
    svc = make_svc(tmp_path, md_at(now), full_llm(), now=now)
    svc.store.set_cash(REAL, 400.0)                      # total loss 600 >= 500
    stage_ideas(svc, [nifty_idea(price=100.0, symbol="GOLDBEES")])
    await pipeline.open_session(svc, now)
    assert svc.limits[REAL].state(now) is LimitState.HARD_STOPPED
    card, _ = svc.notifier.cards[0]
    assert set(card.sizes) == {LEARN}


# ---------------------------------------------------------------- exits

async def test_exits_job_hard_stop_turns_autopilot_off_and_sends_one_recovery(tmp_path):
    now = at(MON, "11:00")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now), llm, now=now)
    set_autopilot(svc.store, True)
    svc.store.set_cash(REAL, 400.0)

    await pipeline.exits_job(svc, now)
    await pipeline.exits_job(svc, now + timedelta(minutes=3))

    assert svc.exits.checks == [now, now + timedelta(minutes=3)]
    assert get_autopilot(svc.store) is False
    assert sum("hard stop" in t.lower() for t in svc.notifier.texts) == 1
    assert len(svc.notifier.recoveries) == 1
    book, stop_id, text = svc.notifier.recoveries[0]
    assert (book, stop_id) == (REAL, 0) and "chased momentum" in text
    assert [c for c in llm.calls if c[1] == "RecoveryPlan"] == [("portfolio_manager", "RecoveryPlan")]
    assert svc.store.get_state("recovery_sent:realistic:0") is True


async def test_overlapping_jobs_write_one_recovery_plan(tmp_path):
    now = at(MON, "10:20")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now), llm, now=now)
    svc.store.set_cash(REAL, 400.0)
    await asyncio.gather(pipeline.exits_job(svc, now), pipeline.intraday(svc, now))
    assert len(svc.notifier.recoveries) == 1
    assert sum(c[1] == "RecoveryPlan" for c in llm.calls) == 1


async def test_exits_job_recovery_falls_back_to_send_without_send_recovery(tmp_path):
    now = at(MON, "11:00")

    class PlainNotifier(RecordingNotifier):
        send_recovery = None

    svc = make_svc(tmp_path, md_at(now), full_llm(), PlainNotifier(), now=now)
    svc.store.set_cash(REAL, 400.0)
    await pipeline.exits_job(svc, now)
    assert any("chased momentum" in t for t in svc.notifier.texts)


async def test_exits_job_daily_pause_alerts_once_and_outside_hours_noop(tmp_path):
    now = at(MON, "11:00")
    svc = make_svc(tmp_path, md_at(now), full_llm(), now=now)
    svc.store.save_equity_snapshot(LEARN, MON - timedelta(days=3), 100000.0)
    svc.store.set_cash(LEARN, 89000.0)                 # daily loss 11,000 >= 10,000; total < 50,000
    await pipeline.exits_job(svc, now)
    await pipeline.exits_job(svc, now)
    assert sum("daily loss limit" in t for t in svc.notifier.texts) == 1
    assert svc.notifier.recoveries == []

    await pipeline.exits_job(svc, at(MON, "15:45"))
    assert len(svc.exits.checks) == 2


async def test_exits_job_no_recovery_call_when_budget_exhausted(tmp_path):
    now = at(MON, "11:00")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now), llm, now=now)
    svc.store.add_ai_usage("claude-opus-5-5", 0, 0, 0, 0, 100.0, now)
    svc.store.set_cash(REAL, 400.0)
    await pipeline.exits_job(svc, now)
    assert llm.calls == []
    assert len(svc.notifier.recoveries) == 1                 # the owner still gets the approve button
    assert "no ai recovery plan" in svc.notifier.recoveries[0][2].lower()


# ---------------------------------------------------------------- postmarket

async def test_postmarket_saves_snapshots_and_summary_has_ai_spend_inr(tmp_path):
    now = at(MON, "15:45")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now), llm, now=now)
    svc.fetch_closes = lambda d: {"NIFTYBEES": 281.0, "GOLDBEES": 101.0}
    svc.store.add_ai_usage("claude-opus-5-5", 0, 0, 0, 0, 1.5, now)
    stage_ideas(svc, [nifty_idea()])

    await pipeline.postmarket(svc, now)

    assert svc.store.equity_snapshot(REAL, MON) == pytest.approx(1000.0)
    assert svc.store.equity_snapshot(LEARN, MON) == pytest.approx(100000.0)
    assert svc.store.get_state("bench_start") == {"day": MON.isoformat(), "price": 281.0}
    summary = svc.notifier.texts[-1]
    assert "₹132.00" in summary                         # US$1.50 at ₹88
    assert "US$1.50" in summary
    assert REAL in summary and LEARN in summary
    assert "Wait for the pullback" in summary
    assert "NIFTYBEES" in summary
    assert ("reviewer", "ReviewOutput") in llm.calls
    assert svc.store.recent_lessons(1) == ["Wait for the pullback"]


async def test_postmarket_bhavcopy_fallback(tmp_path):
    now = at(MON, "15:45")
    svc = make_svc(tmp_path, md_at(now, prices={"NIFTYBEES": 280.0, "GOLDBEES": 110.0}), full_llm(), now=now)
    svc.store.upsert_position(Position(LEARN, "GOLDBEES", 10, 100.0, 90.0, 130.0, now))
    svc.store.set_cash(LEARN, 99000.0)
    svc.fetch_closes = lambda d: {}                                 # bhavcopy unavailable

    await pipeline.postmarket(svc, now)

    expected = 99000.0 + 10 * 110.0 - compute_charges(Side.SELL, 10, 110.0, svc.settings.charges).total
    assert svc.store.equity_snapshot(LEARN, MON) == pytest.approx(expected)
    assert "GOLDBEES" in svc.market_data.history_calls


async def test_postmarket_summary_counts_trades_and_charges(tmp_path):
    now = at(MON, "15:45")
    svc = make_svc(tmp_path, md_at(now), full_llm(), now=now)
    svc.fetch_closes = lambda d: {"NIFTYBEES": 280.0}
    svc.store.save_fill(Fill("o1", LEARN, "NIFTYBEES", Side.BUY, 10, 280.0, 4.25, False, at(MON, "10:00")))
    await pipeline.postmarket(svc, now)
    summary = svc.notifier.texts[-1]
    assert "1 trade" in summary and "₹4.25" in summary


# ---------------------------------------------------------------- intraday

async def test_intraday_skips_ai_when_exhausted(tmp_path):
    now = at(MON, "11:20")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now), llm, now=now)
    svc.store.add_ai_usage("claude-opus-5-5", 0, 0, 0, 0, 100.0, now)
    svc.store.set_state("premarket_prices", {"NIFTYBEES": 250.0})      # a >2% move: the trigger fires
    svc.store.set_state("shortlist", ["NIFTYBEES"])
    stage_ideas(svc, [])
    old = svc.approvals.create(nifty_idea(), {LEARN: 1}, 280.0, now - timedelta(minutes=30))

    await pipeline.intraday(svc, now)

    assert llm.calls == []
    assert len(svc.executor.retries) == 1                        # pending orders still retried
    assert svc.approvals.get(old.id).status == "expired"


async def test_intraday_reruns_pm_on_price_trigger_and_sends_new_cards(tmp_path):
    now = at(MON, "11:20")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now), llm, now=now)
    svc.store.set_state("premarket_prices", {"NIFTYBEES": 270.0})     # +3.7%
    svc.store.set_state("shortlist", ["NIFTYBEES"])
    stage_ideas(svc, [nifty_idea(id="old")])
    svc.store.set_state("acted_ideas", ["old"])

    await pipeline.intraday(svc, now)

    assert ("portfolio_manager", "PMDecision") in llm.calls
    ids = svc.store.get_state("todays_ideas")
    assert len(ids) == 2 and ids[0] == "old"
    assert len(svc.notifier.cards) == 1 and svc.notifier.cards[0][0].idea_id == ids[1]


async def test_intraday_no_trigger_no_ai(tmp_path):
    now = at(MON, "11:20")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now), llm, now=now)
    svc.store.set_state("premarket_prices", {"NIFTYBEES": 279.0})
    svc.store.set_state("shortlist", ["NIFTYBEES"])
    await pipeline.intraday(svc, now)
    assert llm.calls == []


async def test_intraday_high_materiality_news_triggers_pm(tmp_path):
    now = at(MON, "11:20")
    llm = full_llm()
    llm.responses["NewsDigest"] = NewsDigest(market_mood="neutral", market_summary="x", symbols=[
        SymbolNews(symbol="NIFTYBEES", sentiment=0.5, materiality="high", summary="big")])
    svc = make_svc(tmp_path, md_at(now), llm, now=now, feeds=["https://feed"])
    svc.news_parse = lambda url: news_feed("NIFTYBEES sees record inflows")
    svc.store.set_state("news_since", at(MON, "07:00").isoformat())
    svc.store.set_state("premarket_prices", {"NIFTYBEES": 280.0})
    svc.store.set_state("shortlist", ["NIFTYBEES"])
    await pipeline.intraday(svc, now)
    assert [c[1] for c in llm.calls] == ["NewsDigest", "PMDecision"]


async def test_intraday_reduced_budget_runs_ai_at_most_every_two_hours(tmp_path):
    now = at(MON, "11:20")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now), llm, now=now)
    svc.store.add_ai_usage("claude-opus-5-5", 0, 0, 0, 0, 85.0, now)      # REDUCED
    svc.store.set_state("premarket_prices", {"NIFTYBEES": 270.0})
    svc.store.set_state("shortlist", ["NIFTYBEES"])

    await pipeline.intraday(svc, now)
    await pipeline.intraday(svc, now + timedelta(hours=1))
    assert sum(c[1] == "PMDecision" for c in llm.calls) == 1
    await pipeline.intraday(svc, now + timedelta(hours=2))
    assert sum(c[1] == "PMDecision" for c in llm.calls) == 2


# ---------------------------------------------------------------- weekly

async def test_weekly_keeps_fundamentals_when_unavailable_and_sends_golive(tmp_path):
    sat = date(2026, 11, 7)
    now = at(sat, "10:00")
    llm = full_llm()
    svc = make_svc(tmp_path, md_at(now), llm, now=now)
    svc.store.set_state("fundamentals", {"ITC": {"symbol": "ITC", "quality": "ok", "valuation": "fair",
                                                 "notes": "old"}})
    await pipeline.weekly(svc, now)
    text = "\n".join(svc.notifier.texts)
    assert "fundamentals unavailable this week" in text.lower()
    assert svc.store.get_state("fundamentals")["ITC"]["notes"] == "old"
    assert "go-live" in text.lower()
