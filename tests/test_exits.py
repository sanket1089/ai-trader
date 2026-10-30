import logging
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta
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
    no_stop = [x for x in rig.notifier.texts if "no stop-loss" in x]
    assert sorted(no_stop) == sorted([f"{BOOK} {SYMBOL} has no stop-loss set",
                                      f"{LEARNING} {SYMBOL} has no stop-loss set"])


async def test_once_only_alert_is_marked_only_after_it_was_delivered(rig):
    hold(rig.store, stop=0.0, target=300.0)
    price(rig, 285.0, NOW)
    key = f"nostop_alerted:{BOOK}:{SYMBOL}"

    rig.notifier.fail = True                                 # Telegram is down for the first check
    await rig.watcher.check(NOW)
    assert rig.notifier.texts == [] and not rig.store.get_state(key)

    rig.notifier.fail = False                                # next check: tried again, then marked
    await rig.watcher.check(NOW + 3 * MIN)
    assert rig.notifier.texts == [f"{BOOK} {SYMBOL} has no stop-loss set"]
    assert rig.store.get_state(key) is True

    await rig.watcher.check(NOW + 6 * MIN)
    assert len(rig.notifier.texts) == 1


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


async def step(rig, minutes, p):
    """One check ``minutes`` after NOW, with the price at ``p``."""
    later = NOW + minutes * MIN
    price(rig, p, later)
    return await rig.watcher.check(later)


async def test_reprice_after_2min_max_3_then_alert_once(slow_rig):
    rig = slow_rig
    hold(rig.store)
    await step(rig, 0, 268.0)
    [oid] = order_ids(rig.db_path)

    def stored():
        return rig.store.get_order(oid)

    await step(rig, 1, 267.0)                                # 1 minute old: too soon
    assert stored().attempts == 0 and len(rig.broker.calls) == 1

    await step(rig, 2, 265.0)                                # 2 minutes: re-priced at the current price
    assert (stored().limit_price, stored().attempts) == (exit_limit(265.0, 1.0), 1)
    assert datetime.fromisoformat(rig.store.get_state(f"exit_repriced_at:{oid}")) == NOW + 2 * MIN

    await step(rig, 3, 264.0)                                # only 1 minute since the last re-price
    assert stored().attempts == 1 and len(rig.broker.calls) == 2

    await step(rig, 4, 263.0)
    await step(rig, 6, 262.0)
    assert stored().attempts == 3 and rig.notifier.texts == []       # three re-prices, no alert yet

    await step(rig, 8, 261.0)                                # the 4th: past the cap, the owner is told
    assert stored().attempts == 4
    [alert] = rig.notifier.texts
    assert BOOK in alert and SYMBOL in alert and oid in alert
    assert rig.store.get_state(f"exit_alerted:{oid}") is True

    await step(rig, 10, 260.0)                               # and it keeps being re-priced, silently
    await step(rig, 12, 259.0)
    assert stored().attempts == 6 and stored().status is OrderStatus.PENDING
    assert len(rig.notifier.texts) == 1                      # the alert fired exactly once
    assert order_ids(rig.db_path) == [oid]                   # always the same single exit order
    assert [(o.limit_price, o.attempts) for o, _ in rig.broker.calls][1:] == [
        (exit_limit(p, 1.0), n) for n, p in enumerate((265.0, 263.0, 262.0, 261.0, 260.0, 259.0), start=1)]


async def test_exit_keeps_being_tried_after_the_cap_and_fills_when_the_price_recovers(slow_rig):
    rig = slow_rig
    hold(rig.store)
    await step(rig, 0, 268.0)
    [oid] = order_ids(rig.db_path)
    for minutes in (2, 4, 6, 8, 10):                         # 5 re-prices: past the cap of 3
        await step(rig, minutes, 266.0)
    assert rig.store.get_order(oid).attempts == 5 and len(rig.broker.calls) == 6

    rig.broker.fills = True                                  # the price recovers through the limit
    fills = await step(rig, 12, 270.0)

    assert [f.order_id for f in fills] == [oid]
    assert rig.store.get_order(oid).status is OrderStatus.FILLED
    assert len(rig.broker.calls) == 7                        # try_fill was called every time


async def test_a_day_old_pending_exit_is_repriced_the_next_day(slow_rig):
    """Friday's exit never filled and used up its re-prices; Monday's first check re-prices it again
    instead of leaving it at a stale limit (and blocking every new exit for the holding)."""
    rig = slow_rig
    hold(rig.store)
    friday = at(date(2026, 10, 30), "10:00")
    for minutes in (0, 2, 4, 6, 8):                          # created, then 4 re-prices: past the cap
        price(rig, 268.0, friday + minutes * MIN)
        await rig.watcher.check(friday + minutes * MIN)
    [oid] = order_ids(rig.db_path)
    assert rig.store.get_order(oid).attempts == 4

    price(rig, 262.0, NOW)                                   # Monday 10:00
    assert await rig.watcher.check(NOW) == []

    fresh = rig.store.get_order(oid)
    assert (fresh.limit_price, fresh.attempts, fresh.status) == (exit_limit(262.0, 1.0), 5, OrderStatus.PENDING)
    assert order_ids(rig.db_path) == [oid]
    assert len(rig.broker.calls) == 6


async def test_unfilled_exit_alert_is_retried_until_it_was_delivered(slow_rig):
    rig = slow_rig
    hold(rig.store)
    await step(rig, 0, 268.0)
    [oid] = order_ids(rig.db_path)
    for minutes in (2, 4, 6):
        await step(rig, minutes, 266.0)

    rig.notifier.fail = True
    await step(rig, 8, 266.0)                                # past the cap, but Telegram is down
    assert rig.notifier.texts == [] and not rig.store.get_state(f"exit_alerted:{oid}")

    rig.notifier.fail = False
    await step(rig, 10, 266.0)                               # delivered now, and marked
    assert len(rig.notifier.texts) == 1 and rig.store.get_state(f"exit_alerted:{oid}") is True
    await step(rig, 12, 266.0)
    assert len(rig.notifier.texts) == 1


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

    assert order_ids(rig.db_path) == [] and rig.broker.calls == [] and rig.risk.calls == []
    assert any("stale" in r.getMessage() for r in caplog.records)


async def test_unusable_price_is_skipped(rig):
    hold(rig.store)
    price(rig, float("nan"), NOW)
    assert await rig.watcher.check(NOW) == []
    assert order_ids(rig.db_path) == []


BLIND = f"no fresh price for {SYMBOL}; its stop-loss can't be checked"


def blind_alerts(rig):
    return [x for x in rig.notifier.texts if "no fresh price" in x]


@pytest.mark.parametrize("kind", ["missing", "stale", "nan"])
async def test_blind_stop_alerts_the_owner_once_per_day(rig, kind):
    hold(rig.store)

    def feed(now):
        if kind == "stale":
            rig.md.set(SYMBOL, 280.0, now - timedelta(minutes=40))
        elif kind == "nan":
            rig.md.set(SYMBOL, float("nan"), now - MIN)

    feed(NOW)
    await rig.watcher.check(NOW)
    assert blind_alerts(rig) == [BLIND]

    feed(NOW + 3 * MIN)
    await rig.watcher.check(NOW + 3 * MIN)                   # same day: not again
    assert blind_alerts(rig) == [BLIND]

    tomorrow = at(MON + timedelta(days=1), "10:00")
    feed(tomorrow)
    await rig.watcher.check(tomorrow)                        # a new day: told again
    assert blind_alerts(rig) == [BLIND, BLIND]
    assert rig.store.get_state(f"noquote_alerted:{BOOK}:{SYMBOL}:{MON.isoformat()}") is True


async def test_no_blind_stop_alert_while_the_price_is_fresh(rig):
    hold(rig.store)
    price(rig, 285.0, NOW)
    await rig.watcher.check(NOW)
    assert rig.notifier.texts == []


async def test_blind_stop_alert_is_retried_if_the_notifier_was_down(rig):
    hold(rig.store)
    rig.notifier.fail = True
    await rig.watcher.check(NOW)
    rig.notifier.fail = False
    await rig.watcher.check(NOW + 3 * MIN)
    await rig.watcher.check(NOW + 6 * MIN)
    assert blind_alerts(rig) == [BLIND]


# ---------------------------------------------------------------- one bad position must not stop the pass


def errors(rig):
    return [x for x in rig.notifier.texts if "error" in x.lower()]


async def test_one_failing_position_does_not_stop_the_pass_and_the_owner_is_told_once(rig, caplog):
    rig.risk.raise_for = {"GOLDBEES"}                        # sorts first, so it fails before NIFTYBEES
    hold(rig.store, symbol="GOLDBEES")
    hold(rig.store, symbol=SYMBOL)
    price(rig, 250.0, NOW, symbol="GOLDBEES")
    price(rig, 268.0, NOW)

    with caplog.at_level(logging.ERROR, logger="aitrader.exits"):
        fills = await rig.watcher.check(NOW)

    assert [f.symbol for f in fills] == [SYMBOL]             # the other position's stop still sold
    [alert] = errors(rig)
    assert BOOK in alert and "GOLDBEES" in alert and "RuntimeError" in alert
    assert "secret-token-123" not in " ".join(rig.notifier.texts)        # never the exception message
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "RuntimeError" in logged and "GOLDBEES" in logged and BOOK in logged
    assert "secret-token-123" not in logged and not any(r.exc_info for r in caplog.records)

    rig.store.remove_position(BOOK, SYMBOL)                  # NIFTYBEES is sold; GOLDBEES keeps failing
    await rig.watcher.check(NOW + 3 * MIN)
    assert len(errors(rig)) == 1                             # same day: still once
    assert rig.store.get_state(f"check_error_alerted:{BOOK}:GOLDBEES:{MON.isoformat()}") is True

    tomorrow = at(MON + timedelta(days=1), "10:00")
    price(rig, 250.0, tomorrow, symbol="GOLDBEES")
    await rig.watcher.check(tomorrow)
    assert len(errors(rig)) == 2                             # a new day: told again


async def test_a_broker_error_is_contained_and_the_exit_is_picked_up_again(rig):
    rig.broker.raise_for = {SYMBOL}
    hold(rig.store)
    price(rig, 268.0, NOW)

    assert await rig.watcher.check(NOW) == []                # the broker blew up: nothing escapes
    assert len(errors(rig)) == 1
    [oid] = order_ids(rig.db_path)
    assert rig.store.get_order(oid).status is OrderStatus.PENDING     # saved before the broker call

    rig.broker.raise_for = set()                             # the broker is back
    fills = await step(rig, 2, 267.0)
    assert [f.order_id for f in fills] == [oid]              # the same exit is re-priced and fills
    assert order_ids(rig.db_path) == [oid]


async def test_error_alert_is_retried_if_the_notifier_was_down(rig):
    rig.risk.raise_for = {SYMBOL}
    hold(rig.store)
    price(rig, 268.0, NOW)
    rig.notifier.fail = True
    await rig.watcher.check(NOW)
    rig.notifier.fail = False
    await rig.watcher.check(NOW + MIN)
    await rig.watcher.check(NOW + 2 * MIN)
    assert len(errors(rig)) == 1


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
