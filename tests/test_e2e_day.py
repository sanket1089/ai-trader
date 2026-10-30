"""One simulated trading day through the real jobs (offline): premarket, open_session, exits_job every
3 minutes, postmarket, with a scripted quote feed and a fake LLM. Autopilot is ON.

The feed is delayed, as in real life: until 09:40 it still serves the previous close (stale), so the
09:20 open_session cannot act on the idea, and the entry happens from ``exits_job`` as soon as a
fresh quote arrives. Everything between the fakes and the notifier is real: the Store, RiskManager,
Executor (with its real 1-per-second throttle), PaperBroker, ExitWatcher and the ``guarded`` job
wrapper that the scheduler uses.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pytest
from pipeline_fakes import LEARN, REAL, FakeLLM, FakeMarketData, RecordingNotifier, make_svc

from aitrader.agents.schemas import (
    IdeaOut,
    NewsDigest,
    PMDecision,
    ReviewOutput,
    SymbolNews,
    TechnicalBatch,
    TechnicalView,
    TradeReview,
)
from aitrader.app import JOB_FUNCS, guarded
from aitrader.approvals import set_autopilot
from aitrader.calendar import at
from aitrader.charges import compute_charges
from aitrader.models import OrderStatus, Side

MON = date(2026, 11, 2)
PREV_CLOSE_TS = at(date(2026, 10, 30), "15:29")        # Friday's last tick
FEED_DELAY = timedelta(minutes=15)
FEED_LIVE_FROM = at(MON, "09:40")                      # the delayed feed catches up with the open
SLIPPAGE = 0.30 / 100                                  # PaperBroker: 0.05 base + 0.25 for the delayed feed

STOCK = "ITC"
BENCH = "NIFTYBEES"
LIMIT, STOP, TARGET = 400.0, 380.0, 432.0              # the PM's idea, set against the previous close

# (quote time, price) waypoints, linearly interpolated. ITC crosses its 432 target at about 12:32.
PATHS = {
    STOCK: [("09:00", 400.0), ("09:27", 400.0), ("10:30", 404.0), ("11:30", 414.0), ("12:15", 428.0),
            ("12:30", 431.0), ("12:45", 437.0), ("13:30", 425.0), ("15:30", 420.0)],
    BENCH: [("09:00", 280.0), ("12:00", 282.0), ("15:30", 283.5)],
}


class ScriptedFeed(FakeMarketData):
    """``FakeMarketData`` whose quotes follow ``PATHS`` and are delayed by 15 minutes.

    ``advance(now)`` moves the feed to the job time ``now``. Before ``FEED_LIVE_FROM`` it still serves
    Friday's close (a stale quote); after it, the price at ``now - 15 min`` stamped with that time.
    ``daily_history`` ends at the current price, which is what the postmarket job reads as the close.
    """

    def __init__(self):
        super().__init__({s: pts[0][1] for s, pts in PATHS.items()}, PREV_CLOSE_TS)

    @staticmethod
    def price_at(symbol: str, when: datetime) -> float:
        points = [(at(MON, hhmm), price) for hhmm, price in PATHS[symbol]]
        if when <= points[0][0]:
            return points[0][1]
        for (t0, p0), (t1, p1) in zip(points, points[1:]):
            if when <= t1:
                return round(p0 + (p1 - p0) * (when - t0) / (t1 - t0), 2)
        return points[-1][1]

    def advance(self, now: datetime) -> None:
        if now < FEED_LIVE_FROM:
            self.ts = PREV_CLOSE_TS
            self.prices = {s: pts[0][1] for s, pts in PATHS.items()}
        else:
            self.ts = now - FEED_DELAY
            self.prices = {s: self.price_at(s, self.ts) for s in PATHS}


class Clock:
    def __init__(self):
        self.now = at(MON, "08:30")

    def __call__(self) -> datetime:
        return self.now


def pm_llm() -> FakeLLM:
    return FakeLLM({
        "NewsDigest": NewsDigest(market_mood="bullish", market_summary="FMCG in demand", symbols=[
            SymbolNews(symbol=STOCK, sentiment=0.5, materiality="medium", summary="volumes rising")]),
        "TechnicalBatch": TechnicalBatch(views=[TechnicalView(
            symbol=STOCK, trend="up", support=390, resistance=432, signal="buy", confidence=0.7,
            notes="base breakout")]),
        "PMDecision": PMDecision(market_view="constructive", ideas=[IdeaOut(
            symbol=STOCK, side="BUY", limit_price=LIMIT, stop=STOP, target=TARGET, expected_move_pct=8.0,
            confidence=0.7, holding_days=3, reason="ITC breakout with rising volumes")]),
        "ReviewOutput": ReviewOutput(
            lessons=["Entries after the feed catches up (09:40) still reached the target"],
            trade_reviews=[TradeReview(symbol=STOCK, verdict="good", note="target hit mid-day")]),
    })


def news_entry():
    published = time.struct_time((2026, 11, 2, 2, 0, 0, 0, 0, 0))                 # 07:30 IST
    return {"feed": {"title": "F"}, "bozo": 0, "entries": [
        {"title": "ITC and the Nifty open firm as FMCG volumes rise", "link": "https://n/1", "summary": "",
         "published_parsed": published}]}


def job_times() -> list[tuple[datetime, str]]:
    """The day's schedule in time order: premarket, open_session, exits every 3 minutes, postmarket."""
    events = [(at(MON, "08:30"), "premarket"), (at(MON, "09:20"), "open_session"),
              (at(MON, "15:45"), "postmarket")]
    tick = at(MON, "09:15")
    while tick <= at(MON, "15:30"):
        events.append((tick, "exits"))
        tick += timedelta(minutes=3)
    return sorted(events, key=lambda e: e[0])


@dataclass
class Day:
    svc: object
    feed: ScriptedFeed
    notifier: RecordingNotifier
    llm: FakeLLM
    ran: list = field(default_factory=list)                 # (job time, job name), in order
    orders_after_open_session: list | None = None

    @property
    def store(self):
        return self.svc.store


async def simulate_day(db_dir) -> Day:
    clock, feed, notifier, llm = Clock(), ScriptedFeed(), RecordingNotifier(), pm_llm()
    svc = make_svc(db_dir, feed, llm, notifier, spy=False, feeds=["https://feed"], now=clock.now)
    svc.news_parse = lambda url: news_entry()
    set_autopilot(svc.store, True)
    jobs = {name: guarded(svc, name, fn, clock) for name, fn in JOB_FUNCS.items()}

    day = Day(svc, feed, notifier, llm)
    for now, name in job_times():
        clock.now = now
        feed.advance(now)
        await jobs[name]()
        day.ran.append((now, name))
        if name == "open_session":
            day.orders_after_open_session = (svc.store.orders_with_status(OrderStatus.FILLED)
                                             + svc.store.pending_orders())
    return day


@pytest.fixture(scope="module")
def day(tmp_path_factory):
    """The whole day, run once (the real executor throttles orders to one a second)."""
    result = asyncio.run(simulate_day(tmp_path_factory.mktemp("e2e")))
    yield result
    result.store.close()


def fills_of(day, book):
    start = at(MON, "00:00")
    return day.store.fills(book, start, start + timedelta(days=1))


def split(day, book):
    """The book's one entry fill and one exit fill (and nothing else)."""
    fills = fills_of(day, book)
    entries = [f for f in fills if f.side is Side.BUY and not f.is_exit]
    exits = [f for f in fills if f.side is Side.SELL and f.is_exit]
    assert len(fills) == 2, fills
    assert len(entries) == 1 and len(exits) == 1, (book, entries, exits)
    return entries[0], exits[0]


def test_the_day_ran_every_job_without_a_job_error(day):
    names = [name for _, name in day.ran]
    assert names[0] == "premarket" and names[-1] == "postmarket"
    assert names.count("open_session") == 1
    assert names.count("exits") == 126                              # 09:15 to 15:30, every 3 minutes
    texts = day.notifier.texts
    assert not [t for t in texts if "failed" in t.lower() or "error" in t.lower()], texts
    for job in ("premarket", "open_session", "exits", "postmarket"):
        assert day.store.get_state(f"last_run:{job}") is not None, job
    assert [m for _, m in day.llm.calls] == ["NewsDigest", "TechnicalBatch", "PMDecision", "ReviewOutput"]


def test_premarket_produced_the_idea_and_the_09_20_open_was_too_early_to_act(day):
    ids = day.store.get_state("todays_ideas")
    assert len(ids) == 1
    idea = day.store.get_idea(ids[0])
    assert idea.symbol == STOCK and idea.side is Side.BUY and idea.target == TARGET and idea.stop == STOP
    # at 09:20 the delayed feed still served Friday's close: nothing could be placed yet
    assert day.orders_after_open_session == []
    assert day.store.get_state("acted_ideas") == [ids[0]]           # and it was acted on later, once


def test_both_books_bought_once_and_sold_once_at_the_target(day):
    ticks = [now for now, name in day.ran if name == "exits"]
    first_fresh = next(t for t in ticks if t >= FEED_LIVE_FROM)
    first_target = next(t for t in ticks if t >= FEED_LIVE_FROM
                        and day.feed.price_at(STOCK, t - FEED_DELAY) >= TARGET)
    assert first_fresh == at(MON, "09:42")                          # 09:39 is still stale
    assert at(MON, "12:30") < first_target < at(MON, "12:55")       # mid-day

    for book in (REAL, LEARN):
        entry, exit_ = split(day, book)
        assert entry.symbol == exit_.symbol == STOCK
        assert entry.qty == exit_.qty and entry.qty >= 1
        # quote timestamps: the entry used the first fresh quote, the exit the quote that hit the target
        assert entry.ts == first_fresh - FEED_DELAY
        assert exit_.ts == first_target - FEED_DELAY
        # and the orders were placed by those exits_job ticks (not by the 09:20 open_session)
        assert day.store.get_order(entry.order_id).created_at == first_fresh
        assert day.store.get_order(exit_.order_id).created_at == first_target
        # fill prices: quote x (1 +/- 0.30%) to the paisa, from the scripted path itself
        assert entry.price == round(day.feed.price_at(STOCK, entry.ts) * (1 + SLIPPAGE), 2)
        assert exit_.price == round(day.feed.price_at(STOCK, exit_.ts) * (1 - SLIPPAGE), 2)
        assert day.feed.price_at(STOCK, exit_.ts) >= TARGET
        assert "target hit" in day.store.get_order(exit_.order_id).reason
        assert day.store.positions(book) == []
    # sizes: ₹1,000 buys 2 units at about ₹400; the learning book's 20% cap (₹20,000) buys 49
    assert split(day, REAL)[0].qty == 2
    assert split(day, LEARN)[0].qty == 49


def test_cash_is_start_plus_pnl_minus_exact_charges(day):
    capital = {b.name: b.capital for b in day.svc.settings.books}
    for book in (REAL, LEARN):
        entry, exit_ = split(day, book)
        buy_charges = compute_charges(Side.BUY, entry.qty, entry.price, day.svc.settings.charges).total
        sell_charges = compute_charges(Side.SELL, exit_.qty, exit_.price, day.svc.settings.charges).total
        assert entry.charges == pytest.approx(buy_charges, abs=1e-9)
        assert exit_.charges == pytest.approx(sell_charges, abs=1e-9)
        assert sell_charges >= day.svc.settings.charges.dp_per_sell     # the DP charge is on the sell
        pnl = entry.qty * (exit_.price - entry.price)
        expected = capital[book] + pnl - buy_charges - sell_charges
        assert day.store.cash(book) == pytest.approx(expected, abs=1e-6)
        assert pnl > 0 and day.store.cash(book) > capital[book]     # the target was hit: a net win


def test_every_filled_order_went_through_the_risk_gate(day):
    filled = day.store.orders_with_status(OrderStatus.FILLED)
    assert len(filled) == 4                                         # entry + exit in two books
    assert all(day.store.risk_approved(o.id) for o in filled)
    assert day.store.pending_orders() == []
    assert day.store.orders_with_status(OrderStatus.REJECTED) == []
    assert day.store.orders_with_status(OrderStatus.CANCELLED) == []


def test_postmarket_saved_equity_and_sent_the_summary_with_lessons(day):
    for book in (REAL, LEARN):
        assert day.store.equity_snapshot(book, MON) == pytest.approx(day.store.cash(book))   # all sold
    summaries = [t for t in day.notifier.texts if t.startswith("Daily summary")]
    assert len(summaries) == 1
    summary = summaries[0]
    assert f"learning: equity ₹{day.store.cash(LEARN):,.2f}" in summary
    assert f"realistic: equity ₹{day.store.cash(REAL):,.2f}" in summary
    for book in (REAL, LEARN):                                      # each book: its trades and charges paid
        charges = sum(f.charges for f in fills_of(day, book))
        assert f"{book}: equity" in summary
        assert f"2 trade(s), charges paid ₹{charges:,.2f}" in summary, summary
    assert summary.count("2 trade(s)") == 2
    assert "AI spend (a learning cost, apart from trading P&L): today US$0.00" in summary
    assert "NIFTYBEES since 2026-11-02:" in summary and "no close today" not in summary
    assert "Lessons added:" in summary and "Entries after the feed catches up" in summary
    assert day.store.recent_lessons(5) == ["Entries after the feed catches up (09:40) still reached the target"]
    assert day.notifier.cards == []                                 # autopilot ON: no approval cards


def test_the_owner_heard_about_each_fill(day):
    texts = day.notifier.texts
    assert sum(t.startswith("Filled") and " BUY " in t for t in texts) == 2
    assert sum(t.startswith("Filled") and " SELL " in t and "target hit" in t for t in texts) == 2
    assert not [t for t in texts if "HARD STOP" in t or "Rejected" in t]
    assert any(t.startswith("Premarket:") and STOCK in t for t in texts)


def test_quote_helper_is_the_independent_oracle():
    assert ScriptedFeed.price_at(STOCK, at(MON, "09:27")) == 400.0
    assert ScriptedFeed.price_at(STOCK, at(MON, "12:30")) == 431.0
    assert ScriptedFeed.price_at(STOCK, at(MON, "12:33")) == 432.2
