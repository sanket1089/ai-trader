import logging
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import date, timedelta

import pytest

from aitrader.broker import PaperBroker
from aitrader.calendar import at
from aitrader.charges import compute_charges
from aitrader.config import ChargesConfig
from aitrader.models import Fill, Order, OrderStatus, Position, Quote, Side, TradeIdea
from aitrader.store import Store

CFG = ChargesConfig(brokerage_per_order=0.0, stt_pct=0.1, stamp_buy_pct=0.015,
                    exchange_pct=0.00322, sebi_pct=0.0001, gst_pct=18.0, dp_per_sell=15.34)

MON = date(2026, 11, 2)
BOOK = "realistic"
SYMBOL = "NIFTYBEES"
NOW = at(MON, "10:00")


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "t.db"


@pytest.fixture
def store(db_path):
    s = Store(db_path)
    s.set_cash(BOOK, 1000.0)
    s.set_cash("learning", 100000.0)
    s.save_idea(idea("i1", stop=270.0, target=300.0))
    s.save_idea(idea("i2", stop=275.0, target=310.0))
    yield s
    s.close()


@pytest.fixture
def broker(store):
    return PaperBroker(store, CFG)


def idea(id, stop, target):
    return TradeIdea(id=id, symbol=SYMBOL, side=Side.BUY, limit_price=280.0, stop=stop, target=target,
                     expected_move_pct=3.5, confidence=0.7, holding_days=5, reason="trend")


def order(id="o1", side=Side.BUY, qty=3, limit_price=282.0, is_exit=False, idea_id="i1",
          status=OrderStatus.PENDING, book=BOOK, symbol=SYMBOL):
    return Order(id=id, book=book, symbol=symbol, side=side, qty=qty, limit_price=limit_price,
                 reason="test", is_exit=is_exit, idea_id=idea_id, created_at=NOW, status=status)


def quote(price, symbol=SYMBOL, ts=NOW):
    return Quote(symbol=symbol, price=price, ts=ts)


def hold(store, qty=3, avg_price=280.0, stop=270.0, target=300.0, idea_id="i1", book=BOOK):
    store.upsert_position(Position(book=book, symbol=SYMBOL, qty=qty, avg_price=avg_price, stop=stop,
                                   target=target, opened_at=at(MON - timedelta(days=3), "11:00"),
                                   idea_id=idea_id))


def all_fills(store, book=BOOK) -> list[Fill]:
    return store.fills(book, at(MON - timedelta(days=30), "00:00"), at(MON + timedelta(days=30), "00:00"))


def stored_status(db_path, order_id) -> str | None:
    """Order status as persisted, read through a separate connection."""
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute("SELECT status FROM orders WHERE id = ?", (order_id,)).fetchall()
    return rows[0][0] if rows else None


def order_rows(db_path) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]


# ---- BUY ----

def test_buy_fills_with_slippage_and_charges(store, broker, db_path):
    o = order(qty=3, limit_price=282.0)
    f = broker.try_fill(o, quote(280.0))

    assert f is not None
    assert f.price == pytest.approx(280.84)               # 280 x 1.003
    expected_charges = compute_charges(Side.BUY, 3, 280.84, CFG).total
    assert f.charges == pytest.approx(expected_charges)
    assert (f.order_id, f.book, f.symbol, f.side, f.qty, f.is_exit, f.ts) == \
        ("o1", BOOK, SYMBOL, Side.BUY, 3, False, NOW)
    assert store.cash(BOOK) == pytest.approx(1000 - 842.52 - expected_charges)
    assert all_fills(store) == [f]
    assert o.status is OrderStatus.FILLED
    assert stored_status(db_path, "o1") == "FILLED"
    assert store.pending_orders() == []
    (p,) = store.positions(BOOK)
    assert (p.qty, p.stop, p.target, p.idea_id, p.opened_at) == (3, 270.0, 300.0, "i1", NOW)
    assert p.avg_price == pytest.approx(280.84)


def test_buy_does_not_touch_the_other_book(store, broker):
    broker.try_fill(order(), quote(280.0))
    assert store.cash("learning") == 100000.0
    assert store.positions("learning") == []


def test_buy_not_filled_above_limit(store, broker, db_path):
    o = order(limit_price=280.0)                          # would fill at 280.84
    assert broker.try_fill(o, quote(280.0)) is None
    assert o.status is OrderStatus.PENDING
    assert stored_status(db_path, "o1") == "PENDING"      # saved, and still waiting
    assert [x.id for x in store.pending_orders()] == ["o1"]
    assert store.cash(BOOK) == 1000.0
    assert store.positions(BOOK) == []
    assert all_fills(store) == []


def test_buy_fills_when_slipped_price_equals_limit(store, broker):
    f = broker.try_fill(order(limit_price=280.84), quote(280.0))
    assert f is not None and f.price == pytest.approx(280.84)


def test_buy_rejected_insufficient_cash(store, broker, db_path):
    store.set_cash(BOOK, 500.0)
    o = order(qty=3, limit_price=282.0)                   # needs 842.52 + charges
    assert broker.try_fill(o, quote(280.0)) is None
    assert o.status is OrderStatus.REJECTED
    assert stored_status(db_path, "o1") == "REJECTED"
    assert store.pending_orders() == []
    assert store.cash(BOOK) == 500.0
    assert store.positions(BOOK) == []
    assert all_fills(store) == []


def test_buy_rejected_when_cash_covers_value_but_not_charges(store, broker):
    # 3 x 280.84 = 842.52; charges are ~0.99, so 842.60 covers the value but not value + charges.
    store.set_cash(BOOK, 842.60)
    o = order(qty=3, limit_price=282.0)
    assert broker.try_fill(o, quote(280.0)) is None
    assert o.status is OrderStatus.REJECTED
    assert store.cash(BOOK) == 842.60


def test_buy_that_spends_exactly_all_cash_fills(store, broker):
    charges = compute_charges(Side.BUY, 3, 280.84, CFG).total
    store.set_cash(BOOK, 3 * 280.84 + charges)
    assert broker.try_fill(order(), quote(280.0)) is not None
    assert store.cash(BOOK) == pytest.approx(0.0, abs=1e-9)
    assert store.cash(BOOK) >= 0.0


def test_buy_averages_existing_position(store, broker):
    hold(store, qty=3, avg_price=280.0, stop=270.0, target=300.0, idea_id="i1")
    o = order(id="o2", qty=2, limit_price=291.0, idea_id="i2")
    f = broker.try_fill(o, quote(290.0))                  # fills at 290.87

    assert f is not None and f.price == pytest.approx(290.87)
    (p,) = store.positions(BOOK)
    assert p.qty == 5
    assert p.avg_price == pytest.approx((3 * 280.0 + 2 * 290.87) / 5)
    # The newest idea's levels apply to the whole position; the original open time is kept.
    assert (p.stop, p.target, p.idea_id) == (275.0, 310.0, "i2")
    assert p.opened_at == at(MON - timedelta(days=3), "11:00")
    charges = compute_charges(Side.BUY, 2, 290.87, CFG).total
    assert store.cash(BOOK) == pytest.approx(1000 - 2 * 290.87 - charges)


@pytest.mark.parametrize("idea_id", [None, "missing"])
def test_buy_without_idea_gets_zero_levels_and_warns(store, broker, caplog, idea_id):
    caplog.set_level(logging.WARNING, logger="aitrader.broker.paper")
    f = broker.try_fill(order(idea_id=idea_id), quote(280.0))
    assert f is not None
    (p,) = store.positions(BOOK)
    assert (p.qty, p.stop, p.target) == (3, 0.0, 0.0)
    assert any(r.levelno == logging.WARNING and "o1" in r.getMessage() for r in caplog.records)


def test_averaging_without_idea_keeps_existing_levels(store, broker, caplog):
    caplog.set_level(logging.WARNING, logger="aitrader.broker.paper")
    hold(store, qty=3, avg_price=280.0, stop=270.0, target=300.0, idea_id="i1")
    assert broker.try_fill(order(id="o2", qty=1, limit_price=291.0, idea_id=None), quote(290.0))
    (p,) = store.positions(BOOK)
    assert (p.qty, p.stop, p.target, p.idea_id) == (4, 270.0, 300.0, "i1")
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ---- SELL ----

def test_sell_removes_position_and_charges_dp(store, broker, db_path):
    hold(store, qty=3, avg_price=280.0)
    o = order(id="s1", side=Side.SELL, qty=3, limit_price=290.0, is_exit=True)
    f = broker.try_fill(o, quote(300.0))

    assert f is not None
    assert f.price == pytest.approx(299.1)                # 300 x 0.997
    charges = compute_charges(Side.SELL, 3, 299.1, CFG)
    assert charges.dp == 15.34
    assert f.charges == pytest.approx(charges.total)
    assert f.is_exit is True and f.side is Side.SELL
    assert store.cash(BOOK) == pytest.approx(1000 + 3 * 299.1 - charges.total)
    assert store.positions(BOOK) == []
    assert o.status is OrderStatus.FILLED
    assert stored_status(db_path, "s1") == "FILLED"
    assert all_fills(store) == [f]


def test_sell_partial_reduces_position_and_keeps_levels(store, broker):
    hold(store, qty=5, avg_price=280.0, stop=270.0, target=300.0, idea_id="i1")
    f = broker.try_fill(order(id="s1", side=Side.SELL, qty=2, limit_price=290.0, idea_id=None),
                        quote(300.0))
    assert f is not None
    (p,) = store.positions(BOOK)
    assert (p.qty, p.avg_price, p.stop, p.target, p.idea_id) == (3, 280.0, 270.0, 300.0, "i1")
    assert p.opened_at == at(MON - timedelta(days=3), "11:00")


def test_sell_not_filled_below_limit(store, broker, db_path):
    hold(store, qty=3)
    o = order(id="s1", side=Side.SELL, limit_price=305.0)  # would fill at 299.10
    assert broker.try_fill(o, quote(300.0)) is None
    assert o.status is OrderStatus.PENDING
    assert stored_status(db_path, "s1") == "PENDING"
    assert store.positions(BOOK)[0].qty == 3
    assert store.cash(BOOK) == 1000.0
    assert all_fills(store) == []


def test_sell_fills_when_slipped_price_equals_limit(store, broker):
    hold(store, qty=3)
    f = broker.try_fill(order(id="s1", side=Side.SELL, limit_price=299.1), quote(300.0))
    assert f is not None and f.price == pytest.approx(299.1)


def test_sell_more_than_held_rejected(store, broker, db_path):
    hold(store, qty=3)
    o = order(id="s1", side=Side.SELL, qty=5, limit_price=290.0)
    assert broker.try_fill(o, quote(300.0)) is None
    assert o.status is OrderStatus.REJECTED
    assert stored_status(db_path, "s1") == "REJECTED"
    assert store.positions(BOOK)[0].qty == 3
    assert store.cash(BOOK) == 1000.0
    assert all_fills(store) == []


def test_sell_with_no_position_rejected(store, broker):
    o = order(id="s1", side=Side.SELL, qty=1, limit_price=290.0)
    assert broker.try_fill(o, quote(300.0)) is None
    assert o.status is OrderStatus.REJECTED
    assert store.positions(BOOK) == []


def test_sell_rejected_when_it_would_overdraw_cash(store, broker):
    # 1 x 9.97 of proceeds cannot pay the 15.34 DP charge, and cash is 5: cash would go negative.
    store.set_cash(BOOK, 5.0)
    hold(store, qty=1, avg_price=10.0, stop=9.0, target=12.0)
    o = order(id="s1", side=Side.SELL, qty=1, limit_price=9.0)
    assert broker.try_fill(o, quote(10.0)) is None
    assert o.status is OrderStatus.REJECTED
    assert store.cash(BOOK) == 5.0
    assert store.positions(BOOK)[0].qty == 1


def test_sell_with_proceeds_below_dp_charge_fills_when_cash_covers_it(store, broker):
    store.set_cash(BOOK, 50.0)
    hold(store, qty=1, avg_price=10.0, stop=9.0, target=12.0)
    f = broker.try_fill(order(id="s1", side=Side.SELL, qty=1, limit_price=9.0), quote(10.0))
    assert f is not None
    assert 0.0 <= store.cash(BOOK) < 50.0


# ---- invalid orders and non-pending orders ----

@pytest.mark.parametrize("qty", [0, -2])
def test_qty_below_one_rejected(store, broker, db_path, qty):
    o = order(qty=qty)
    assert broker.try_fill(o, quote(280.0)) is None
    assert o.status is OrderStatus.REJECTED
    assert stored_status(db_path, "o1") == "REJECTED"
    assert store.cash(BOOK) == 1000.0
    assert store.positions(BOOK) == []


@pytest.mark.parametrize("status", [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED])
def test_try_fill_on_non_pending_order_changes_nothing(store, broker, db_path, status):
    o = order(status=status)
    assert broker.try_fill(o, quote(280.0)) is None
    assert o.status is status
    assert order_rows(db_path) == 0                       # not even saved
    assert store.cash(BOOK) == 1000.0
    assert store.positions(BOOK) == []
    assert all_fills(store) == []


def test_filled_order_is_never_filled_twice(store, broker):
    o = order()
    assert broker.try_fill(o, quote(280.0)) is not None
    cash_after = store.cash(BOOK)
    assert broker.try_fill(o, quote(280.0)) is None
    assert len(all_fills(store)) == 1
    assert store.cash(BOOK) == cash_after
    assert store.positions(BOOK)[0].qty == 3


def test_order_saved_by_caller_is_updated_not_duplicated(store, broker, db_path):
    o = order()
    store.save_order(o)                                   # the Executor saves before try_fill
    assert broker.try_fill(o, quote(280.0)) is not None
    assert order_rows(db_path) == 1
    assert stored_status(db_path, "o1") == "FILLED"


def test_stale_pending_copy_of_a_filled_order_cannot_fill_again(store, broker, db_path):
    o = order()
    stale_copy = replace(o)                               # still says PENDING after the original fills
    assert broker.try_fill(o, quote(280.0)) is not None
    cash_after = store.cash(BOOK)
    with pytest.raises(sqlite3.IntegrityError):
        broker.try_fill(stale_copy, quote(280.0))
    assert len(all_fills(store)) == 1
    assert store.cash(BOOK) == cash_after
    assert stored_status(db_path, "o1") == "FILLED"


def test_unfilled_order_can_be_retried_and_then_fills(store, broker, db_path):
    o = order(limit_price=281.0)
    assert broker.try_fill(o, quote(281.0)) is None       # 281.84 > 281: persisted as PENDING
    assert stored_status(db_path, "o1") == "PENDING"
    f = broker.try_fill(o, quote(279.0))                  # retry on the stored order: 279.84 <= 281
    assert f is not None and f.price == pytest.approx(279.84)
    assert order_rows(db_path) == 1
    assert stored_status(db_path, "o1") == "FILLED"
    assert len(all_fills(store)) == 1


def test_quote_for_a_different_symbol_is_ignored(store, broker, db_path):
    o = order()
    assert broker.try_fill(o, quote(280.0, symbol="GOLDBEES")) is None
    assert o.status is OrderStatus.PENDING
    assert store.cash(BOOK) == 1000.0
    assert store.positions(BOOK) == []


@pytest.mark.parametrize("price", [0.0, -5.0, float("nan")])
def test_unusable_quote_price_leaves_order_pending(store, broker, price):
    o = order()
    assert broker.try_fill(o, quote(price)) is None
    assert o.status is OrderStatus.PENDING
    assert store.cash(BOOK) == 1000.0
    assert all_fills(store) == []


# ---- slippage ----

def test_slippage_is_configurable(store):
    f = PaperBroker(store, CFG, slippage_pct=0.05).try_fill(order(), quote(280.0))
    assert f is not None and f.price == pytest.approx(280.14)    # 280 x 1.0005


def test_default_slippage_is_030_percent(store):
    broker = PaperBroker(store, CFG)
    buy = broker.try_fill(order(id="b1", limit_price=282.0), quote(280.0))
    assert buy is not None and buy.price == pytest.approx(280.84)    # 280 x (1 + 0.30%)
    sell = broker.try_fill(order(id="s1", side=Side.SELL, qty=2, limit_price=290.0), quote(300.0))
    assert sell is not None and sell.price == pytest.approx(299.1)   # 300 x (1 - 0.30%)


@pytest.mark.parametrize("bad", [-0.1, 100.0, 250.0])
def test_slippage_out_of_range_rejected(store, bad):
    with pytest.raises(ValueError):
        PaperBroker(store, CFG, slippage_pct=bad)


# ---- atomicity ----

def _boom(*args, **kwargs):
    raise RuntimeError("disk on fire")


@pytest.mark.parametrize("failing_write", ["save_fill", "update_order", "upsert_position", "set_cash"])
def test_buy_fill_is_atomic(store, broker, db_path, monkeypatch, failing_write):
    hold(store, qty=3, avg_price=280.0, stop=270.0, target=300.0, idea_id="i1")
    positions_before = store.positions(BOOK)
    o = order(id="o2", qty=2, limit_price=291.0, idea_id="i2")
    store.save_order(o)

    monkeypatch.setattr(store, failing_write, _boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        broker.try_fill(o, quote(290.0))
    monkeypatch.undo()

    assert all_fills(store) == []
    assert stored_status(db_path, "o2") == "PENDING"
    assert o.status is OrderStatus.PENDING                # the caller's object still matches the DB
    assert store.positions(BOOK) == positions_before
    assert store.cash(BOOK) == 1000.0
    # and the order is still fillable afterwards
    assert broker.try_fill(o, quote(290.0)) is not None


@pytest.mark.parametrize("failing_write", ["save_fill", "update_order", "remove_position", "set_cash"])
def test_sell_fill_is_atomic(store, broker, db_path, monkeypatch, failing_write):
    hold(store, qty=3, avg_price=280.0)
    positions_before = store.positions(BOOK)
    o = order(id="s1", side=Side.SELL, qty=3, limit_price=290.0, is_exit=True)
    store.save_order(o)

    monkeypatch.setattr(store, failing_write, _boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        broker.try_fill(o, quote(300.0))
    monkeypatch.undo()

    assert all_fills(store) == []
    assert stored_status(db_path, "s1") == "PENDING"
    assert o.status is OrderStatus.PENDING
    assert store.positions(BOOK) == positions_before
    assert store.cash(BOOK) == 1000.0


def test_failed_fill_of_unsaved_order_leaves_it_pending(store, broker, db_path, monkeypatch):
    o = order()
    monkeypatch.setattr(store, "set_cash", _boom)
    with pytest.raises(RuntimeError):
        broker.try_fill(o, quote(280.0))
    monkeypatch.undo()
    assert stored_status(db_path, "o1") == "PENDING"      # saved first, so it can be retried
    assert all_fills(store) == []
    assert store.positions(BOOK) == []


# ---- cancel ----

def test_cancel_pending_order(store, broker, db_path):
    o = order(limit_price=280.0)
    assert broker.try_fill(o, quote(280.0)) is None       # stays PENDING
    broker.cancel("o1")
    assert stored_status(db_path, "o1") == "CANCELLED"
    assert store.pending_orders() == []
    # a cancelled order can no longer fill, even with the stored copy
    assert broker.try_fill(replace(o, status=OrderStatus.CANCELLED), quote(270.0)) is None
    assert all_fills(store) == []


def test_cancel_leaves_filled_and_unknown_orders_alone(store, broker, db_path):
    assert broker.try_fill(order(), quote(280.0)) is not None
    broker.cancel("o1")                                   # already FILLED
    broker.cancel("nope")                                 # never existed
    assert stored_status(db_path, "o1") == "FILLED"
    assert order_rows(db_path) == 1
