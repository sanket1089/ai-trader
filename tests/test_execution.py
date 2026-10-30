import asyncio
import json
import logging
import sqlite3
from contextlib import closing
from datetime import date, timedelta

import pytest
from exec_fakes import FakeBroker, FakeNotifier, FakeRisk, FakeTime

from aitrader.broker import PaperBroker
from aitrader.calendar import at
from aitrader.config import ChargesConfig
from aitrader.execution import Executor, marketable_limit
from aitrader.models import Order, OrderStatus, Position, Quote, Side, TradeIdea
from aitrader.store import Store

MON = date(2026, 11, 2)
YESTERDAY = date(2026, 10, 30)
BOOK = "realistic"
SYMBOL = "NIFTYBEES"
NOW = at(MON, "10:00")
CHARGES = ChargesConfig(brokerage_per_order=0.0, stt_pct=0.1, stamp_buy_pct=0.015, exchange_pct=0.00322,
                        sebi_pct=0.0001, gst_pct=18.0, dp_per_sell=15.34)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "t.db"


@pytest.fixture
def store(db_path):
    s = Store(db_path)
    s.set_cash(BOOK, 1000.0)
    s.set_cash("learning", 100000.0)
    s.save_idea(idea())
    yield s
    s.close()


@pytest.fixture
def t():
    return FakeTime()


@pytest.fixture
def notifier():
    return FakeNotifier()


def idea():
    return TradeIdea(id="i1", symbol=SYMBOL, side=Side.BUY, limit_price=280.0, stop=270.0, target=300.0,
                     expected_move_pct=8.0, confidence=0.7, holding_days=5, reason="trend")


def order(id="o1", side=Side.BUY, qty=3, limit=280.0, is_exit=False, book=BOOK, symbol=SYMBOL,
          created_at=NOW, idea_id="i1", reason="trend"):
    return Order(id=id, book=book, symbol=symbol, side=side, qty=qty, limit_price=limit, reason=reason,
                 is_exit=is_exit, idea_id=idea_id, created_at=created_at)


def quote(price=280.0, symbol=SYMBOL):
    return Quote(symbol=symbol, price=price, ts=NOW - timedelta(minutes=1))


def executor(store, broker, risk, notifier, t, dry_run=False, **kw):
    return Executor(store, broker, risk, notifier, dry_run=dry_run, sleep=t.sleep, clock=t.clock, **kw)


def risk_rows(db_path) -> list[tuple[str, int, list[str]]]:
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute("SELECT order_id, approved, reasons FROM risk_log ORDER BY id").fetchall()
    return [(r[0], r[1], json.loads(r[2])) for r in rows]


def hold(store, book=BOOK, symbol=SYMBOL, qty=3):
    store.upsert_position(Position(book=book, symbol=symbol, qty=qty, avg_price=280.0, stop=270.0,
                                   target=300.0, opened_at=at(YESTERDAY, "11:00"), idea_id="i1"))


# ---------------------------------------------------------------- submit


async def test_rejected_order_notifies_reasons(store, db_path, notifier, t):
    broker = FakeBroker(store, fills=True)
    risk = FakeRisk(approve=False, reasons=["outside entry window 09:20–15:10", "quote is stale"])
    ex = executor(store, broker, risk, notifier, t)

    assert await ex.submit(order(), quote(), NOW, idea()) is None

    assert broker.calls == []
    assert store.get_order("o1").status is OrderStatus.REJECTED
    [text] = notifier.texts
    assert "outside entry window 09:20–15:10" in text and "quote is stale" in text
    assert BOOK in text and SYMBOL in text
    assert risk_rows(db_path) == [("o1", 0, ["outside entry window 09:20–15:10", "quote is stale"])]


async def test_approved_order_fills_and_notifies(store, db_path, notifier, t):
    broker = FakeBroker(store, fills=True)
    risk = FakeRisk()
    ex = executor(store, broker, risk, notifier, t)
    q, i = quote(), idea()

    fill = await ex.submit(order(), q, NOW, i)

    assert fill is not None and fill.order_id == "o1"
    assert store.get_order("o1").status is OrderStatus.FILLED
    assert risk.calls[0][1:] == (q, NOW, i)                  # the risk gate saw the quote, time and idea
    assert risk_rows(db_path) == [("o1", 1, [])]             # approvals are recorded too
    [text] = notifier.texts
    for part in (BOOK, "BUY", "3", SYMBOL, "₹280.00", "₹1.50"):
        assert part in text


async def test_order_is_saved_pending_before_the_broker_is_called(store, notifier, t):
    seen = []

    class WatchingBroker(FakeBroker):
        def try_fill(self, order, quote):
            seen.append(store.get_order(order.id).status)
            return super().try_fill(order, quote)

    ex = executor(store, WatchingBroker(store), FakeRisk(), notifier, t)
    await ex.submit(order(), quote(), NOW, idea())
    assert seen == [OrderStatus.PENDING]


async def test_unfilled_order_stays_pending(store, notifier, t):
    ex = executor(store, FakeBroker(store, fills=False), FakeRisk(), notifier, t)
    assert await ex.submit(order(), quote(), NOW, idea()) is None
    assert store.get_order("o1").status is OrderStatus.PENDING
    assert notifier.texts == []


async def test_broker_rejection_is_reported(store, notifier, t):
    ex = executor(store, FakeBroker(store, rejects=True), FakeRisk(), notifier, t)
    assert await ex.submit(order(), quote(), NOW, idea()) is None
    assert store.get_order("o1").status is OrderStatus.REJECTED
    [text] = notifier.texts
    assert "rejected by the broker" in text and SYMBOL in text


async def test_dry_run_never_calls_broker(store, db_path, notifier, t):
    broker = FakeBroker(store, fills=True)
    ex = executor(store, broker, FakeRisk(), notifier, t, dry_run=True)

    assert await ex.submit(order(reason="trend up"), quote(), NOW, idea()) is None

    assert broker.calls == [] and broker.cancelled == []
    saved = store.get_order("o1")
    assert saved.status is OrderStatus.CANCELLED
    assert saved.reason.startswith("trend up") and "dry-run" in saved.reason     # appended, not replaced
    [text] = notifier.texts
    assert "dry-run" in text.lower() and SYMBOL in text and BOOK in text
    assert risk_rows(db_path) == [("o1", 1, [])]


async def test_dry_run_still_reports_a_risk_rejection_as_a_rejection(store, notifier, t):
    broker = FakeBroker(store)
    ex = executor(store, broker, FakeRisk(approve=False, reasons=["quote is stale"]), notifier, t,
                  dry_run=True)
    await ex.submit(order(), quote(), NOW, idea())
    assert store.get_order("o1").status is OrderStatus.REJECTED
    assert "quote is stale" in notifier.texts[0]


async def test_throttle_spaces_orders_1s(store, notifier, t):
    broker = FakeBroker(store)
    ex = executor(store, broker, FakeRisk(), notifier, t)

    for i in range(3):
        await ex.submit(order(id=f"o{i}"), quote(), NOW, idea())
    assert t.sleeps == [1.0, 1.0]                      # nothing before the first, 1 s before each next

    t.sleeps.clear()
    t.now += 0.4                                        # 0.4 s since the last order: wait the other 0.6 s
    await ex.submit(order(id="o3"), quote(), NOW, idea())
    assert t.sleeps == [pytest.approx(0.6)]

    t.sleeps.clear()
    t.now += 5.0                                        # long enough ago: no wait
    await ex.submit(order(id="o4"), quote(), NOW, idea())
    assert t.sleeps == []


async def test_throttle_applies_to_rejected_orders_too(store, notifier, t):
    ex = executor(store, FakeBroker(store), FakeRisk(approve=False, reasons=["no"]), notifier, t)
    await ex.submit(order(id="a"), quote(), NOW, idea())
    await ex.submit(order(id="b"), quote(), NOW, idea())
    assert t.sleeps == [1.0]


async def test_concurrent_submits_are_serialised(store, notifier, t):
    reached = []

    class ClockBroker(FakeBroker):
        def try_fill(self, order, quote):
            reached.append((order.id, t.now))
            return super().try_fill(order, quote)

    ex = executor(store, ClockBroker(store), FakeRisk(), notifier, t)
    await ex.submit(order(id="warm"), quote(), NOW, idea())
    await asyncio.gather(ex.submit(order(id="a"), quote(), NOW, idea()),
                         ex.submit(order(id="b"), quote(), NOW, idea()))
    # the second waits for the first, so the orders reach the broker a full second apart
    assert reached == [("warm", 1000.0), ("a", 1001.0), ("b", 1002.0)]


async def test_notifier_failure_never_breaks_trading(store, notifier, t):
    notifier.fail = True
    ex = executor(store, FakeBroker(store, fills=True), FakeRisk(), notifier, t)
    fill = await ex.submit(order(), quote(), NOW, idea())
    assert fill is not None
    assert store.get_order("o1").status is OrderStatus.FILLED


async def test_submit_with_the_real_paper_broker_updates_cash_and_position(store, notifier, t):
    ex = executor(store, PaperBroker(store, CHARGES), FakeRisk(), notifier, t)
    fill = await ex.submit(order(limit=282.0), quote(280.0), NOW, idea())
    assert fill is not None
    assert store.get_order("o1").status is OrderStatus.FILLED
    assert [(p.symbol, p.qty, p.stop, p.target) for p in store.positions(BOOK)] == [(SYMBOL, 3, 270.0, 300.0)]
    assert store.cash(BOOK) < 1000.0
    assert f"₹{fill.price:.2f}" in notifier.texts[0] and f"₹{fill.charges:.2f}" in notifier.texts[0]


# ---------------------------------------------------------------- marketable_limit


@pytest.mark.parametrize("price, buy, sell", [(280.0, 281.40, 278.60), (284.0, 285.42, 282.58),
                                              (401.13, 403.13, 399.13), (10.01, 10.06, 9.96)])
def test_marketable_limit_is_the_edge_of_the_band_to_the_paisa(price, buy, sell):
    assert marketable_limit(Side.BUY, price, 0.5) == buy        # floored: never outside the band
    assert marketable_limit(Side.SELL, price, 0.5) == sell      # ceiled: never outside the band


@pytest.mark.parametrize("price", [1.0, 9.99, 99.99, 280.0, 401.13, 1234.57, 2999.95])
def test_marketable_limit_fills_at_the_paper_broker_inside_the_risk_band(price):
    # A limit equal to the last price never fills (the broker adds 0.30% slippage), and one more than
    # 0.5% away is refused by the risk gate: the marketable limit must satisfy both.
    buy, sell = marketable_limit(Side.BUY, price, 0.5), marketable_limit(Side.SELL, price, 0.5)
    assert round(price * 1.003, 2) <= buy <= price * 1.005 + 1e-6
    assert round(price * 0.997, 2) >= sell >= price * 0.995 - 1e-6


# ---------------------------------------------------------------- submit_exit / reprice_exit


async def test_exit_cancels_a_pending_discretionary_sell_first(store, notifier, t):
    hold(store, qty=3)
    hold(store, symbol="GOLDBEES", qty=3)
    store.save_order(order(id="sell-mine", side=Side.SELL))                       # reserves the holding
    store.save_order(order(id="sell-other-symbol", side=Side.SELL, symbol="GOLDBEES"))
    store.save_order(order(id="sell-other-book", side=Side.SELL, book="learning"))
    store.save_order(order(id="buy-same-symbol", side=Side.BUY))
    broker = FakeBroker(store)
    ex = executor(store, broker, FakeRisk(), notifier, t)

    ex_order = order(id="exit1", side=Side.SELL, is_exit=True, limit=277.2, reason="stop hit at ₹280.00")
    await ex.submit_exit(ex_order, quote(280.0), NOW)

    assert broker.cancelled == ["sell-mine"]
    cancelled = store.get_order("sell-mine")
    assert cancelled.status is OrderStatus.CANCELLED and cancelled.reason.startswith("trend")
    for untouched in ("sell-other-symbol", "sell-other-book", "buy-same-symbol"):
        assert store.get_order(untouched).status is OrderStatus.PENDING
    assert [o.id for o, _ in broker.calls] == ["exit1"]


async def test_exit_is_never_a_second_exit_while_one_is_pending(store, notifier, t):
    hold(store)
    broker = FakeBroker(store)
    ex = executor(store, broker, FakeRisk(), notifier, t)
    first = order(id="exit1", side=Side.SELL, is_exit=True, limit=277.2)
    await ex.submit_exit(first, quote(), NOW)

    second = order(id="exit2", side=Side.SELL, is_exit=True, limit=277.2)
    assert await ex.submit_exit(second, quote(), NOW) is None

    assert store.get_order("exit2") is None
    assert [o.id for o, _ in broker.calls] == ["exit1"]


async def test_submit_exit_refuses_anything_but_an_exit_sell(store, notifier, t):
    ex = executor(store, FakeBroker(store), FakeRisk(), notifier, t)
    with pytest.raises(ValueError):
        await ex.submit_exit(order(side=Side.SELL, is_exit=False), quote(), NOW)
    with pytest.raises(ValueError):
        await ex.submit_exit(order(side=Side.BUY, is_exit=True), quote(), NOW)


async def test_exit_in_dry_run_cancels_nothing_and_never_calls_the_broker(store, notifier, t):
    hold(store)
    store.save_order(order(id="sell-mine", side=Side.SELL))
    broker = FakeBroker(store, fills=True)
    ex = executor(store, broker, FakeRisk(), notifier, t, dry_run=True)

    await ex.submit_exit(order(id="exit1", side=Side.SELL, is_exit=True, limit=277.2), quote(), NOW)

    assert broker.calls == [] and broker.cancelled == []
    assert store.get_order("sell-mine").status is OrderStatus.PENDING
    assert store.get_order("exit1").status is OrderStatus.CANCELLED


async def test_reprice_exit_updates_the_same_order_and_retries_the_fill(store, db_path, notifier, t):
    hold(store)
    broker = FakeBroker(store)
    risk = FakeRisk()
    ex = executor(store, broker, risk, notifier, t)
    await ex.submit_exit(order(id="exit1", side=Side.SELL, is_exit=True, limit=277.2), quote(280.0), NOW)

    later = NOW + timedelta(minutes=2)
    assert await ex.reprice_exit("exit1", 270.0, quote(272.0), later) is None

    fresh = store.get_order("exit1")
    assert (fresh.limit_price, fresh.attempts, fresh.status) == (270.0, 1, OrderStatus.PENDING)
    assert [(o.limit_price, o.attempts) for o, _ in broker.calls] == [(277.2, 0), (270.0, 1)]
    assert len(risk.calls) == 2                                   # the retry went through the risk gate
    assert [r[0] for r in risk_rows(db_path)] == ["exit1", "exit1"]


async def test_reprice_exit_fills_and_notifies(store, notifier, t):
    hold(store)
    broker = FakeBroker(store)
    ex = executor(store, broker, FakeRisk(), notifier, t)
    await ex.submit_exit(order(id="exit1", side=Side.SELL, is_exit=True, limit=277.2,
                               reason="stop hit at ₹280.00"), quote(280.0), NOW)
    broker.fills = True

    fill = await ex.reprice_exit("exit1", 268.0, quote(270.0), NOW + timedelta(minutes=2))

    assert fill is not None and store.get_order("exit1").status is OrderStatus.FILLED
    assert "SELL" in notifier.texts[0] and "stop hit" in notifier.texts[0]


async def test_reprice_exit_ignores_an_order_that_is_no_longer_pending(store, notifier, t):
    """A caller's copy may be stale: only the stored order counts, and a finished one is left alone."""
    hold(store)
    broker = FakeBroker(store)
    ex = executor(store, broker, FakeRisk(), notifier, t)
    done = order(id="exit1", side=Side.SELL, is_exit=True, limit=277.2)
    done.status = OrderStatus.FILLED
    store.save_order(done)

    assert await ex.reprice_exit("exit1", 270.0, quote(272.0), NOW) is None
    assert await ex.reprice_exit("missing", 270.0, quote(272.0), NOW) is None

    assert broker.calls == []
    assert store.get_order("exit1").attempts == 0


async def test_reprice_exit_in_dry_run_does_nothing(store, notifier, t):
    hold(store)
    store.save_order(order(id="exit1", side=Side.SELL, is_exit=True, limit=277.2))
    broker = FakeBroker(store, fills=True)
    ex = executor(store, broker, FakeRisk(), notifier, t, dry_run=True)
    assert await ex.reprice_exit("exit1", 270.0, quote(272.0), NOW) is None
    assert broker.calls == [] and store.get_order("exit1").attempts == 0


# ---------------------------------------------------------------- retry_pending


async def test_retry_cancels_entries_left_from_earlier_days(store, notifier, t):
    old = at(YESTERDAY, "14:00")
    hold(store)
    store.save_order(order(id="old-buy", created_at=old))
    store.save_order(order(id="old-sell", side=Side.SELL, created_at=old))
    store.save_order(order(id="old-exit", side=Side.SELL, is_exit=True, created_at=old))
    store.save_order(order(id="today-buy", symbol="GOLDBEES"))
    broker = FakeBroker(store)
    ex = executor(store, broker, FakeRisk(), notifier, t)

    assert await ex.retry_pending({}, NOW) == []

    for gone in ("old-buy", "old-sell"):
        o = store.get_order(gone)
        assert o.status is OrderStatus.CANCELLED and o.reason.startswith("trend")
    assert sorted(broker.cancelled) == ["old-buy", "old-sell"]
    assert store.get_order("old-exit").status is OrderStatus.PENDING        # exits belong to the ExitWatcher
    assert store.get_order("today-buy").status is OrderStatus.PENDING
    assert len(notifier.texts) == 2 and all("cancel" in x.lower() for x in notifier.texts)


async def test_retry_uses_the_ist_date_for_staleness(store, notifier, t):
    """23:50 IST yesterday is still yesterday; 00:10 IST today is today."""
    store.save_order(order(id="late", created_at=at(YESTERDAY, "23:50")))
    store.save_order(order(id="early", created_at=at(MON, "00:10"), symbol="GOLDBEES"))
    ex = executor(store, FakeBroker(store), FakeRisk(), notifier, t)
    await ex.retry_pending({}, NOW)
    assert store.get_order("late").status is OrderStatus.CANCELLED
    assert store.get_order("early").status is OrderStatus.PENDING


async def test_retry_fills_todays_pending_orders_through_risk_and_broker(store, db_path, notifier, t):
    store.save_order(order(id="buy1"))
    broker = FakeBroker(store, fills=True)
    risk = FakeRisk()
    ex = executor(store, broker, risk, notifier, t)
    q = quote(279.0)

    fills = await ex.retry_pending({SYMBOL: q}, NOW)

    assert [f.order_id for f in fills] == ["buy1"]
    assert store.get_order("buy1").status is OrderStatus.FILLED
    sent_order, sent_quote, sent_now, sent_idea = risk.calls[0]
    assert (sent_order.id, sent_quote, sent_now, sent_idea) == ("buy1", q, NOW, idea())   # idea from the store
    assert risk_rows(db_path) == [("buy1", 1, [])]
    assert "BUY" in notifier.texts[0] and SYMBOL in notifier.texts[0]


async def test_retry_works_from_the_stored_order_not_a_stale_copy(store, notifier, t):
    """Once an order is filled the next retry cannot see it as pending (a stale PENDING copy would
    make the real broker raise an IntegrityError)."""
    broker = PaperBroker(store, CHARGES)
    ex = executor(store, broker, FakeRisk(), notifier, t)
    stale = order(id="buy1", limit=282.0)
    assert await ex.submit(stale, quote(280.0), NOW, idea()) is not None
    stale.status = OrderStatus.PENDING                                  # the caller's stale copy

    assert await ex.retry_pending({SYMBOL: quote(280.0)}, NOW) == []
    assert len(store.positions(BOOK)) == 1 and store.positions(BOOK)[0].qty == 3


async def test_retry_leaves_an_order_pending_when_risk_now_rejects_it(store, db_path, notifier, t):
    store.save_order(order(id="buy1"))
    broker = FakeBroker(store, fills=True)
    ex = executor(store, broker, FakeRisk(approve=False, reasons=["quote is stale"]), notifier, t)

    assert await ex.retry_pending({SYMBOL: quote()}, NOW) == []

    assert broker.calls == []
    assert store.get_order("buy1").status is OrderStatus.PENDING
    assert risk_rows(db_path) == [("buy1", 0, ["quote is stale"])]      # the decision is still recorded
    assert notifier.texts == []                                         # no message on every retry


async def test_retry_skips_orders_with_no_quote(store, notifier, t):
    store.save_order(order(id="buy1"))
    risk = FakeRisk()
    broker = FakeBroker(store, fills=True)
    ex = executor(store, broker, risk, notifier, t)

    assert await ex.retry_pending({"GOLDBEES": quote(symbol="GOLDBEES")}, NOW) == []

    assert risk.calls == [] and broker.calls == []
    assert store.get_order("buy1").status is OrderStatus.PENDING


async def test_retry_throttles_between_orders(store, notifier, t):
    store.save_order(order(id="a"))
    store.save_order(order(id="b", symbol="GOLDBEES"))
    ex = executor(store, FakeBroker(store), FakeRisk(), notifier, t)
    await ex.retry_pending({SYMBOL: quote(), "GOLDBEES": quote(symbol="GOLDBEES")}, NOW)
    assert t.sleeps == [1.0]


async def test_retry_is_a_no_op_in_dry_run(store, notifier, t):
    store.save_order(order(id="old", created_at=at(YESTERDAY, "14:00")))
    store.save_order(order(id="today", symbol="GOLDBEES"))
    broker = FakeBroker(store, fills=True)
    ex = executor(store, broker, FakeRisk(), notifier, t, dry_run=True)

    assert await ex.retry_pending({SYMBOL: quote(), "GOLDBEES": quote(symbol="GOLDBEES")}, NOW) == []

    assert broker.calls == [] and broker.cancelled == []
    assert store.get_order("old").status is OrderStatus.PENDING


async def test_retry_and_submit_share_one_lock(store, notifier, t):
    """A retry started while a submit is waiting out the throttle queues behind it."""
    store.save_order(order(id="pending1", symbol="GOLDBEES"))
    broker = FakeBroker(store)
    ex = executor(store, broker, FakeRisk(), notifier, t)
    await ex.submit(order(id="warm"), quote(), NOW, idea())

    await asyncio.gather(ex.submit(order(id="a"), quote(), NOW, idea()),
                         ex.retry_pending({"GOLDBEES": quote(symbol="GOLDBEES")}, NOW))

    assert [o.id for o, _ in broker.calls] == ["warm", "a", "pending1"]
    assert t.sleeps == [1.0, 1.0]


async def test_notify_swallows_a_failing_notifier_and_reports_whether_it_got_through(store, notifier, t):
    notifier.fail = True
    ex = executor(store, FakeBroker(store), FakeRisk(), notifier, t)
    assert await ex.notify("hello") is False           # must not raise
    notifier.fail = False
    assert await ex.notify("hello") is True
    assert notifier.texts == ["hello"]


async def test_notify_once_marks_the_key_only_after_the_message_was_delivered(store, notifier, t):
    ex = executor(store, FakeBroker(store), FakeRisk(), notifier, t)

    notifier.fail = True
    assert await ex.notify_once("k", "first") is False
    assert store.get_state("k") is None                # not marked: nobody was told yet

    notifier.fail = False
    assert await ex.notify_once("k", "first") is True  # so it is tried again
    assert store.get_state("k") is True
    assert await ex.notify_once("k", "first") is False # and now it is once
    assert notifier.texts == ["first"]


async def test_notify_once_sends_again_when_the_key_was_re_armed(store, notifier, t):
    ex = executor(store, FakeBroker(store), FakeRisk(), notifier, t)
    await ex.notify_once("k", "x")
    store.set_state("k", False)
    assert await ex.notify_once("k", "x") is True
    assert notifier.texts == ["x", "x"]


# ---------------------------------------------------------------- one bad order must not stop the pass


async def test_retry_carries_on_when_one_order_blows_up_and_alerts_once_per_day(store, notifier, t, caplog):
    store.save_order(order(id="bad", symbol="GOLDBEES"))
    store.save_order(order(id="good", symbol=SYMBOL))
    broker = FakeBroker(store, fills=True)
    ex = executor(store, broker, FakeRisk(raise_for={"GOLDBEES"}), notifier, t)
    quotes = {SYMBOL: quote(), "GOLDBEES": quote(symbol="GOLDBEES")}

    with caplog.at_level(logging.ERROR, logger="aitrader.execution"):
        fills = await ex.retry_pending(quotes, NOW)

    assert [f.order_id for f in fills] == ["good"]                        # the rest of the pass ran
    assert store.get_order("good").status is OrderStatus.FILLED
    assert store.get_order("bad").status is OrderStatus.PENDING           # left to the next retry
    errors = [x for x in notifier.texts if "error" in x.lower()]
    assert len(errors) == 1 and "GOLDBEES" in errors[0] and BOOK in errors[0] and "bad" in errors[0]
    assert "RuntimeError" in errors[0]
    assert "secret-token-123" not in " ".join(notifier.texts)             # never the exception message
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "RuntimeError" in logged and "GOLDBEES" in logged and BOOK in logged
    assert "secret-token-123" not in logged and not any(r.exc_info for r in caplog.records)

    # the same failure again, same day: still not alerted twice
    await ex.retry_pending(quotes, NOW + timedelta(hours=1))
    assert len([x for x in notifier.texts if "error" in x.lower()]) == 1
    assert store.get_state(f"check_error_alerted:{BOOK}:GOLDBEES:{MON.isoformat()}") is True

    # the next day it is told again
    next_day = at(MON + timedelta(days=1), "10:00")
    store.save_order(order(id="bad2", symbol="GOLDBEES", created_at=at(MON + timedelta(days=1), "09:30")))
    await ex.retry_pending(quotes, next_day)
    assert len([x for x in notifier.texts if "error" in x.lower()]) == 2


async def test_retry_error_alert_is_retried_if_the_notifier_was_down(store, notifier, t):
    store.save_order(order(id="bad", symbol="GOLDBEES"))
    ex = executor(store, FakeBroker(store), FakeRisk(raise_for={"GOLDBEES"}), notifier, t)
    quotes = {"GOLDBEES": quote(symbol="GOLDBEES")}

    notifier.fail = True
    await ex.retry_pending(quotes, NOW)
    notifier.fail = False
    await ex.retry_pending(quotes, NOW)
    await ex.retry_pending(quotes, NOW)

    assert len([x for x in notifier.texts if "error" in x.lower()]) == 1


# ---------------------------------------------------------------- cancel_order (the /stop kill switch)


def test_cancel_order_cancels_a_pending_order_at_the_broker_and_in_the_store(store, notifier, t):
    broker = FakeBroker(store)
    ex = executor(store, broker, FakeRisk(), notifier, t)
    store.save_order(order(id="p1"))

    assert ex.cancel_order("p1", "cancelled: /stop") is True

    assert broker.cancelled == ["p1"]
    stored = store.get_order("p1")
    assert stored.status is OrderStatus.CANCELLED
    assert stored.reason == "trend; cancelled: /stop"


def test_cancel_order_ignores_missing_and_closed_orders(store, notifier, t):
    broker = FakeBroker(store)
    ex = executor(store, broker, FakeRisk(), notifier, t)
    done = order(id="done")
    done.status = OrderStatus.FILLED
    store.save_order(done)

    assert ex.cancel_order("nope", "why") is False
    assert ex.cancel_order("done", "why") is False
    assert broker.cancelled == []
    assert store.get_order("done").status is OrderStatus.FILLED
