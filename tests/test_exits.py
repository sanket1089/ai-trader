import logging
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from exec_fakes import FakeBroker, FakeMarketData, FakeNotifier, FakeRisk, FakeTime

from aitrader.broker import PaperBroker
from aitrader.calendar import at
from aitrader.config import load_settings
from aitrader.execution import Executor
from aitrader.exits import ExitWatcher, exit_limit
from aitrader.models import Order, OrderStatus, Position, Side, TradeIdea
from aitrader.risk.limits import LimitState, LossLimits
from aitrader.risk.manager import RiskManager
from aitrader.store import Store

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
SETTINGS = load_settings(REPO_CONFIG, env={"TELEGRAM_OWNER_CHAT_ID": "42", "TELEGRAM_BOT_TOKEN": "x"})
RISK_CFG = SETTINGS.risk

MON = date(2026, 11, 2)                 # a Monday and a trading day
BOOK = "realistic"
LEARNING = "learning"
SYMBOL = "NIFTYBEES"
NOW = at(MON, "10:00")
MIN = timedelta(minutes=1)


def hold(store, book=BOOK, symbol=SYMBOL, qty=3, stop=270.0, target=300.0):
    store.upsert_position(Position(book=book, symbol=symbol, qty=qty, avg_price=280.0, stop=stop,
                                   target=target, opened_at=at(MON - timedelta(days=2), "11:00")))


def order_ids(db_path) -> list[str]:
    with closing(sqlite3.connect(db_path)) as conn:
        return [r[0] for r in conn.execute("SELECT id FROM orders ORDER BY created_at, rowid")]


def make_rig(tmp_path, *, fills=True, books=(BOOK, LEARNING)):
    db_path = tmp_path / "t.db"
    store = Store(db_path)
    store.set_cash(BOOK, 1000.0)
    store.set_cash(LEARNING, 100000.0)
    t, notifier, md = FakeTime(), FakeNotifier(), FakeMarketData()
    broker, risk = FakeBroker(store, fills=fills), FakeRisk()
    executor = Executor(store, broker, risk, notifier, dry_run=False, sleep=t.sleep, clock=t.clock)
    watcher = ExitWatcher(store, executor, md, RISK_CFG, list(books))
    return SimpleNamespace(db_path=db_path, store=store, t=t, notifier=notifier, md=md, broker=broker,
                           risk=risk, executor=executor, watcher=watcher)


@pytest.fixture
def rig(tmp_path):
    r = make_rig(tmp_path)
    yield r
    r.store.close()


@pytest.fixture
def slow_rig(tmp_path):
    """Exit orders never fill, so they stay pending."""
    r = make_rig(tmp_path, fills=False)
    yield r
    r.store.close()


def price(rig, p, at_time, symbol=SYMBOL):
    rig.md.set(symbol, p, at_time - MIN)


# ---------------------------------------------------------------- exit_limit


def test_exit_limit_is_the_rounded_price_through_the_band():
    assert exit_limit(268.0, 1.0) == 265.32
    assert exit_limit(305.0, 1.0) == 301.95
    assert exit_limit(100.0, 1.0) == 99.0


@pytest.mark.parametrize("paise", range(25000, 25200))
def test_exit_limit_is_never_below_what_the_risk_gate_accepts(paise):
    """Plain round() lands up to half a paisa under the band floor for about half of all prices,
    and the risk gate would then refuse the exit. The limit steps up one paisa in that case."""
    p = paise / 100
    floor = p * (1 - 1.0 / 100)
    limit = exit_limit(p, 1.0)
    assert limit >= floor - 1e-6
    assert limit - floor < 0.01                         # and never a whole paisa more than needed


# ---------------------------------------------------------------- triggers


async def test_stop_hit_creates_exit(rig):
    hold(rig.store, qty=3, stop=270.0, target=300.0)
    price(rig, 268.0, NOW)

    fills = await rig.watcher.check(NOW)

    assert len(fills) == 1
    [(sent, q)] = rig.broker.calls
    assert (sent.side, sent.qty, sent.is_exit, sent.book, sent.symbol) == (Side.SELL, 3, True, BOOK, SYMBOL)
    assert sent.limit_price == 265.32                        # 1% through the last price
    assert sent.reason == "stop hit at ₹268.00"
    assert len(sent.id) == 32 and int(sent.id, 16) >= 0      # uuid4 hex
    assert rig.store.get_order(sent.id).status is OrderStatus.FILLED
    assert rig.risk.calls[0][0].id == sent.id                # the exit went through the risk gate
    assert fills[0].order_id == sent.id and fills[0].is_exit


async def test_target_hit_creates_exit(rig):
    hold(rig.store, qty=3, stop=270.0, target=300.0)
    price(rig, 305.0, NOW)

    fills = await rig.watcher.check(NOW)

    assert len(fills) == 1
    [(sent, _)] = rig.broker.calls
    assert sent.limit_price == 301.95
    assert sent.reason == "target hit at ₹305.00"
    assert sent.is_exit and sent.side is Side.SELL


async def test_price_between_stop_and_target_does_nothing(rig):
    hold(rig.store, stop=270.0, target=300.0)
    price(rig, 285.0, NOW)
    assert await rig.watcher.check(NOW) == []
    assert order_ids(rig.db_path) == [] and rig.notifier.texts == []


@pytest.mark.parametrize("p, reason", [(270.0, "stop hit at ₹270.00"), (300.0, "target hit at ₹300.00")])
async def test_stop_and_target_levels_are_inclusive(rig, p, reason):
    hold(rig.store, stop=270.0, target=300.0)
    price(rig, p, NOW)
    assert len(await rig.watcher.check(NOW)) == 1
    assert rig.broker.calls[0][0].reason == reason


async def test_both_books_are_checked_and_only_the_watched_books(tmp_path):
    rig = make_rig(tmp_path, books=(BOOK,))
    hold(rig.store, book=BOOK)
    hold(rig.store, book=LEARNING)
    price(rig, 268.0, NOW)

    fills = await rig.watcher.check(NOW)

    assert [(f.book, f.symbol) for f in fills] == [(BOOK, SYMBOL)]
    assert rig.store.positions(LEARNING)[0].qty == 3
    rig.store.close()


async def test_every_book_holding_the_symbol_exits(rig):
    hold(rig.store, book=BOOK)
    hold(rig.store, book=LEARNING, qty=10)
    price(rig, 268.0, NOW)

    fills = await rig.watcher.check(NOW)

    assert sorted((f.book, f.qty) for f in fills) == [(LEARNING, 10), (BOOK, 3)] or \
        sorted((f.book, f.qty) for f in fills) == [(BOOK, 3), (LEARNING, 10)]
    assert rig.md.calls == [[SYMBOL]]                       # one quote request for both books


async def test_check_with_no_positions_never_asks_for_quotes(rig):
    assert await rig.watcher.check(NOW) == []
    assert rig.md.calls == []


# ---------------------------------------------------------------- unset levels


async def test_unset_levels_never_exit_and_the_missing_stop_is_alerted_once(rig):
    hold(rig.store, stop=0.0, target=0.0)
    for p in (1.0, 280.0, 100000.0):                         # nothing triggers, whatever the price
        price(rig, p, NOW)
        assert await rig.watcher.check(NOW) == []
    assert order_ids(rig.db_path) == [] and rig.broker.calls == []
    assert rig.notifier.texts == [f"{BOOK} {SYMBOL} has no stop-loss set"]       # once, not per check
    assert rig.store.get_state(f"nostop_alerted:{BOOK}:{SYMBOL}") is True


async def test_unset_target_does_not_sell_but_the_stop_still_works(rig):
    hold(rig.store, stop=270.0, target=0.0)
    price(rig, 500.0, NOW)                                   # a target of 0 would "always" be hit
    assert await rig.watcher.check(NOW) == []
    assert rig.notifier.texts == []                          # a stop is set: nothing to warn about

    price(rig, 268.0, NOW)
    assert len(await rig.watcher.check(NOW)) == 1


async def test_unset_stop_does_not_sell_but_the_target_still_works(rig):
    hold(rig.store, stop=0.0, target=300.0)
    price(rig, 1.0, NOW)                                     # a stop of 0 is "unset", never hit
    assert await rig.watcher.check(NOW) == []

    price(rig, 305.0, NOW)
    fills = await rig.watcher.check(NOW)
    assert len(fills) == 1
    assert rig.notifier.texts.count(f"{BOOK} {SYMBOL} has no stop-loss set") == 1


async def test_missing_stop_alert_is_per_position_and_survives_a_missing_quote(rig):
    hold(rig.store, book=BOOK, stop=0.0, target=300.0)
    hold(rig.store, book=LEARNING, stop=0.0, target=300.0)
    # no quote at all for the symbol: the alert does not depend on it
    await rig.watcher.check(NOW)
    await rig.watcher.check(NOW)
    assert sorted(rig.notifier.texts) == sorted([f"{BOOK} {SYMBOL} has no stop-loss set",
                                                 f"{LEARNING} {SYMBOL} has no stop-loss set"])


async def test_missing_stop_alert_rearms_once_a_stop_is_set(rig):
    hold(rig.store, stop=0.0, target=300.0)
    price(rig, 285.0, NOW)
    await rig.watcher.check(NOW)
    hold(rig.store, stop=270.0, target=300.0)                # fixed
    await rig.watcher.check(NOW)
    hold(rig.store, stop=0.0, target=300.0)                  # broken again
    await rig.watcher.check(NOW)
    assert len(rig.notifier.texts) == 2


# ---------------------------------------------------------------- pending exits


async def test_no_duplicate_exit_when_pending(slow_rig):
    rig = slow_rig
    hold(rig.store)
    price(rig, 268.0, NOW)
    assert await rig.watcher.check(NOW) == []
    [first] = order_ids(rig.db_path)
    assert rig.store.get_order(first).status is OrderStatus.PENDING

    price(rig, 266.0, NOW + MIN)                             # still below the stop, one minute later
    assert await rig.watcher.check(NOW + MIN) == []

    assert order_ids(rig.db_path) == [first]                 # no second exit order
    assert len(rig.broker.calls) == 1                        # and not even another attempt yet
    assert rig.store.get_order(first).attempts == 0


async def test_reprice_after_2min_max_3_then_alert_once(slow_rig):
    rig = slow_rig
    hold(rig.store)
    price(rig, 268.0, NOW)
    await rig.watcher.check(NOW)
    [oid] = order_ids(rig.db_path)

    # 1 minute old: too soon
    price(rig, 267.0, NOW + 1 * MIN)
    await rig.watcher.check(NOW + 1 * MIN)
    assert rig.store.get_order(oid).attempts == 0 and len(rig.broker.calls) == 1

    # then every 2 minutes: reprice at the current price, three times
    expected = []
    for n, minutes in enumerate((2, 4, 6), start=1):
        later = NOW + minutes * MIN
        p = 266.0 - n
        price(rig, p, later)
        await rig.watcher.check(later)
        stored = rig.store.get_order(oid)
        expected.append((exit_limit(p, 1.0), n))
        assert (stored.limit_price, stored.attempts, stored.status) == (expected[-1] + (OrderStatus.PENDING,))
        assert order_ids(rig.db_path) == [oid]               # always the same order
    assert [(o.limit_price, o.attempts) for o, _ in rig.broker.calls][1:] == expected
    assert rig.notifier.texts == []                          # no alert yet

    # the fourth time it is left pending and the owner is told, once
    for minutes in (8, 10, 12):
        later = NOW + minutes * MIN
        price(rig, 260.0, later)
        await rig.watcher.check(later)
    stored = rig.store.get_order(oid)
    assert stored.attempts == 3 and stored.status is OrderStatus.PENDING
    assert len(rig.broker.calls) == 4                        # initial try + 3 reprices
    [alert] = rig.notifier.texts
    assert BOOK in alert and SYMBOL in alert and oid in alert
    assert rig.store.get_state(f"exit_alerted:{oid}") is True


async def test_reprice_that_fills_returns_the_fill_and_stops_repricing(slow_rig):
    rig = slow_rig
    hold(rig.store)
    price(rig, 268.0, NOW)
    await rig.watcher.check(NOW)
    rig.broker.fills = True

    later = NOW + 2 * MIN
    price(rig, 267.0, later)
    fills = await rig.watcher.check(later)

    assert len(fills) == 1 and fills[0].order_id == order_ids(rig.db_path)[0]
    assert any("Filled" in x and "SELL" in x for x in rig.notifier.texts)


async def test_a_pending_discretionary_sell_does_not_block_the_exit(rig):
    hold(rig.store, qty=3)
    rig.store.save_order(Order(id="manual", book=BOOK, symbol=SYMBOL, side=Side.SELL, qty=3,
                               limit_price=290.0, reason="take profit", is_exit=False, idea_id=None,
                               created_at=NOW - 10 * MIN))
    price(rig, 268.0, NOW)

    fills = await rig.watcher.check(NOW)

    assert len(fills) == 1
    assert rig.store.get_order("manual").status is OrderStatus.CANCELLED
    assert rig.broker.cancelled == ["manual"]


# ---------------------------------------------------------------- the window


async def test_gap_down_open_exits_at_0915(rig):
    hold(rig.store, stop=270.0, target=300.0)
    open_ = at(MON, "09:15")
    price(rig, 250.0, open_)                                 # gapped far below the stop

    assert await rig.watcher.check(at(MON, "09:14")) == []   # not open for exits yet
    assert rig.md.calls == [] and order_ids(rig.db_path) == []

    fills = await rig.watcher.check(open_)                   # the first check at 09:15 sells
    assert len(fills) == 1
    [(sent, _)] = rig.broker.calls
    assert sent.limit_price == 247.5 and sent.reason == "stop hit at ₹250.00"


async def test_checks_after_the_close_do_nothing(rig):
    hold(rig.store)
    price(rig, 250.0, at(MON, "15:31"))
    assert await rig.watcher.check(at(MON, "15:31")) == []
    assert rig.md.calls == [] and rig.broker.calls == []

    price(rig, 250.0, at(MON, "15:30"))                      # the close itself is still in the window
    assert len(await rig.watcher.check(at(MON, "15:30"))) == 1


async def test_check_needs_an_aware_time(rig):
    hold(rig.store)
    with pytest.raises(ValueError):
        await rig.watcher.check(NOW.replace(tzinfo=None))


# ---------------------------------------------------------------- bad data


async def test_position_without_a_quote_is_skipped_with_a_warning(rig, caplog):
    hold(rig.store, symbol="GOLDBEES")
    hold(rig.store, symbol=SYMBOL)
    price(rig, 268.0, NOW)                                   # only NIFTYBEES has a quote

    with caplog.at_level(logging.WARNING, logger="aitrader.exits"):
        fills = await rig.watcher.check(NOW)

    assert [f.symbol for f in fills] == [SYMBOL]
    assert any("GOLDBEES" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


async def test_stale_quote_is_skipped_so_the_risk_gate_is_not_spammed(rig, caplog):
    hold(rig.store)
    rig.md.set(SYMBOL, 250.0, NOW - timedelta(minutes=26))   # older than 15 + 10 minutes

    with caplog.at_level(logging.WARNING, logger="aitrader.exits"):
        assert await rig.watcher.check(NOW) == []

    assert order_ids(rig.db_path) == [] and rig.notifier.texts == []
    assert any("stale" in r.getMessage() for r in caplog.records)


async def test_unusable_price_is_skipped(rig):
    hold(rig.store)
    price(rig, float("nan"), NOW)
    assert await rig.watcher.check(NOW) == []
    assert order_ids(rig.db_path) == []


# ---------------------------------------------------------------- with the real risk gate


@pytest.fixture
def real(tmp_path):
    """Real store, risk manager and paper broker; fake market data, notifier and clock."""
    db_path = tmp_path / "t.db"
    store = Store(db_path)
    store.set_cash(BOOK, 1000.0)
    store.set_cash(LEARNING, 100000.0)
    limits = {b.name: LossLimits(store, b, SETTINGS.holidays) for b in SETTINGS.books}
    risk = RiskManager(store, SETTINGS, limits, {SYMBOL, "GOLDBEES"})
    t, notifier, md = FakeTime(), FakeNotifier(), FakeMarketData()
    executor = Executor(store, PaperBroker(store, SETTINGS.charges), risk, notifier, dry_run=False,
                        sleep=t.sleep, clock=t.clock)
    watcher = ExitWatcher(store, executor, md, SETTINGS.risk, [BOOK, LEARNING])
    yield SimpleNamespace(store=store, limits=limits, risk=risk, t=t, notifier=notifier, md=md,
                          executor=executor, watcher=watcher, db_path=db_path)
    store.close()


async def test_exit_runs_when_hard_stopped(real):
    hold(real.store, qty=3, stop=270.0, target=300.0)
    assert real.limits[BOOK].evaluate(400.0, NOW).state is LimitState.HARD_STOPPED   # total loss 600 >= 500
    price(real, 268.0, NOW)

    # the stop is real: a new entry is refused...
    idea = TradeIdea(id="i1", symbol=SYMBOL, side=Side.BUY, limit_price=268.0, stop=260.0, target=290.0,
                     expected_move_pct=8.0, confidence=0.7, holding_days=5, reason="trend")
    real.store.save_idea(idea)
    entry = Order(id="entry1", book=BOOK, symbol=SYMBOL, side=Side.BUY, qty=1, limit_price=268.0,
                  reason="trend", is_exit=False, idea_id="i1", created_at=NOW)
    assert await real.executor.submit(entry, real.md.by_symbol[SYMBOL], NOW, idea) is None
    assert real.store.get_order("entry1").status is OrderStatus.REJECTED
    assert "hard stop" in real.notifier.texts[0]

    # ...but the stop-loss exit still sells
    fills = await real.watcher.check(NOW)
    assert len(fills) == 1 and fills[0].is_exit and fills[0].qty == 3
    assert real.store.positions(BOOK) == []
    assert real.store.cash(BOOK) > 1000.0
    assert real.store.risk_approved(fills[0].order_id)
    assert any("Filled" in x and "SELL" in x for x in real.notifier.texts)


async def test_exit_passes_the_real_risk_gate_at_awkward_prices(real):
    """250.51 x 0.99 = 248.0049: a limit of round(.., 2) = 248.00 would be refused as outside the band."""
    hold(real.store, qty=3, stop=260.0, target=300.0)
    price(real, 250.51, NOW)

    fills = await real.watcher.check(NOW)

    assert len(fills) == 1
    assert real.store.get_order(fills[0].order_id).limit_price == 248.01


async def test_a_pending_sell_is_cancelled_so_the_real_risk_gate_approves_the_exit(real):
    hold(real.store, qty=3, stop=270.0, target=300.0)
    real.store.save_order(Order(id="manual", book=BOOK, symbol=SYMBOL, side=Side.SELL, qty=3,
                                limit_price=290.0, reason="take profit", is_exit=False, idea_id=None,
                                created_at=NOW - 10 * MIN))
    price(real, 268.0, NOW)

    fills = await real.watcher.check(NOW)

    assert len(fills) == 1
    assert real.store.get_order("manual").status is OrderStatus.CANCELLED
    assert real.store.positions(BOOK) == []
