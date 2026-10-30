import json
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from aitrader.calendar import IST, at
from aitrader.models import Fill, Order, OrderStatus, Position, Side, TradeIdea
from aitrader.store import Store

MON = date(2026, 11, 2)
TUE = date(2026, 11, 3)
WED = date(2026, 11, 4)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def order(id="o1", book="learning", symbol="NIFTYBEES", side=Side.BUY, qty=3, limit_price=280.0,
          is_exit=False, status=OrderStatus.PENDING, created_at=None, idea_id="i1", attempts=0):
    return Order(id=id, book=book, symbol=symbol, side=side, qty=qty, limit_price=limit_price,
                 reason="test", is_exit=is_exit, idea_id=idea_id,
                 created_at=created_at or at(MON, "10:00"), status=status, attempts=attempts)


def fill(order_id="o1", book="learning", symbol="NIFTYBEES", side=Side.BUY, qty=3, price=280.0,
         charges=1.0, is_exit=False, ts=None):
    return Fill(order_id=order_id, book=book, symbol=symbol, side=side, qty=qty, price=price,
                charges=charges, is_exit=is_exit, ts=ts or at(MON, "10:00"))


def pos(book="learning", symbol="NIFTYBEES", qty=3, avg_price=280.0, stop=270.0, target=300.0):
    return Position(book=book, symbol=symbol, qty=qty, avg_price=avg_price, stop=stop, target=target,
                    opened_at=at(MON, "10:00"), idea_id="i1")


def idea(id="i1"):
    return TradeIdea(id=id, symbol="NIFTYBEES", side=Side.BUY, limit_price=280.0, stop=270.0,
                     target=300.0, expected_move_pct=3.5, confidence=0.7, holding_days=5,
                     reason="trend")


def raw_rows(path, sql):
    """Read rows through a separate connection so tests don't touch Store internals."""
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


# ---- persistence ----

def test_state_survives_reopen(tmp_path):
    s = Store(tmp_path / "t.db")
    s.set_state("autopilot", True)
    s.save_order(order())
    s.upsert_position(pos())
    s2 = Store(tmp_path / "t.db")
    assert s2.get_state("autopilot") is True
    assert len(s2.pending_orders()) == 1
    assert s2.positions("learning")[0].symbol == "NIFTYBEES"


def test_uses_wal_journal_mode(tmp_path):
    s = Store(tmp_path / "t.db")
    assert raw_rows(tmp_path / "t.db", "PRAGMA journal_mode") == [("wal",)]
    s.close()


def test_creates_missing_parent_directory(tmp_path):
    s = Store(tmp_path / "data" / "sub" / "t.db")
    s.set_state("k", 1)
    assert s.get_state("k") == 1
    s.close()


# ---- rehydration: real enums and aware datetimes ----

def test_order_rehydrates_enums_and_aware_datetime(store):
    store.save_order(order(side=Side.SELL, is_exit=True, status=OrderStatus.PENDING))
    o = store.pending_orders()[0]
    assert o.side is Side.SELL
    assert o.status is OrderStatus.PENDING
    assert isinstance(o.side, Side) and isinstance(o.status, OrderStatus)
    assert o.created_at.utcoffset() == timedelta(hours=5, minutes=30)
    assert o.created_at == at(MON, "10:00")
    assert o.is_exit is True and o.idea_id == "i1" and o.attempts == 0 and o.qty == 3


def test_fill_rehydrates_enums_and_aware_datetime(store):
    store.save_fill(fill(side=Side.SELL, is_exit=True))
    f = store.fills("learning", at(MON, "00:00"), at(TUE, "00:00"))[0]
    assert f.side is Side.SELL
    assert isinstance(f.side, Side)
    assert f.ts.utcoffset() == timedelta(hours=5, minutes=30)
    assert f.ts == at(MON, "10:00")
    assert f.is_exit is True and f.charges == 1.0 and f.price == 280.0


def test_position_and_idea_rehydrate(store):
    store.upsert_position(pos())
    p = store.positions("learning")[0]
    assert p.opened_at.utcoffset() == timedelta(hours=5, minutes=30)
    assert p.opened_at == at(MON, "10:00") and p.idea_id == "i1"
    store.save_idea(idea())
    i = store.get_idea("i1")
    assert i == idea()
    assert isinstance(i.side, Side)


def test_datetimes_stored_as_iso_strings_with_offset(tmp_path):
    s = Store(tmp_path / "t.db")
    s.save_order(order())
    s.close()
    raw = raw_rows(tmp_path / "t.db", "SELECT created_at FROM orders")[0][0]
    assert raw.endswith("+05:30")
    assert datetime.fromisoformat(raw) == at(MON, "10:00")


def test_other_timezone_is_normalised_to_ist_on_write(store):
    utc_ts = datetime(2026, 11, 2, 4, 30, tzinfo=timezone.utc)      # 10:00 IST
    store.save_fill(fill(ts=utc_ts))
    f = store.fills("learning", at(MON, "09:59"), at(MON, "10:01"))
    assert len(f) == 1 and f[0].ts == at(MON, "10:00") and f[0].ts.tzinfo == IST


def test_naive_datetime_rejected(store):
    with pytest.raises(ValueError):
        store.save_fill(fill(ts=datetime(2026, 11, 2, 10, 0)))
    with pytest.raises(ValueError):
        store.save_order(order(created_at=datetime(2026, 11, 2, 10, 0)))


# ---- orders and fills ----

def test_pending_orders_filters_by_book_and_status(store):
    store.save_order(order("o1", book="learning"))
    store.save_order(order("o2", book="realistic"))
    store.save_order(order("o3", book="learning", status=OrderStatus.FILLED))
    assert {o.id for o in store.pending_orders()} == {"o1", "o2"}
    assert [o.id for o in store.pending_orders("learning")] == ["o1"]
    assert [o.id for o in store.pending_orders("realistic")] == ["o2"]


def test_update_order_persists_status_attempts_and_price(tmp_path):
    s = Store(tmp_path / "t.db")
    o = order()
    s.save_order(o)
    o.status = OrderStatus.FILLED
    o.attempts = 2
    o.limit_price = 281.5
    s.update_order(o)
    assert s.pending_orders() == []
    s.close()
    rows = raw_rows(tmp_path / "t.db", "SELECT status, attempts, limit_price FROM orders")
    assert rows == [("FILLED", 2, 281.5)]


def test_update_order_unknown_id_raises(store):
    with pytest.raises(KeyError):
        store.update_order(order("missing"))


def test_save_order_duplicate_id_raises(store):
    store.save_order(order())
    with pytest.raises(sqlite3.IntegrityError):
        store.save_order(order())


def test_fills_range_is_start_inclusive_end_exclusive_and_per_book(store):
    store.save_fill(fill(ts=at(MON, "10:00")))
    store.save_fill(fill(ts=at(MON, "11:00")))
    store.save_fill(fill(ts=at(MON, "12:00")))
    store.save_fill(fill(book="realistic", ts=at(MON, "11:00")))
    got = store.fills("learning", at(MON, "10:00"), at(MON, "12:00"))
    assert [f.ts for f in got] == [at(MON, "10:00"), at(MON, "11:00")]


def test_fills_with_fractional_seconds_order_correctly(store):
    base = at(MON, "10:00")
    store.save_fill(fill(order_id="b", ts=base + timedelta(microseconds=500)))
    store.save_fill(fill(order_id="a", ts=base))
    got = store.fills("learning", base, base + timedelta(seconds=1))
    assert [f.order_id for f in got] == ["a", "b"]


# ---- positions and cash ----

def test_upsert_position_replaces_and_remove(store):
    store.upsert_position(pos(qty=3))
    store.upsert_position(pos(qty=5, avg_price=281.0))
    store.upsert_position(pos(book="realistic"))
    got = store.positions("learning")
    assert len(got) == 1 and got[0].qty == 5 and got[0].avg_price == 281.0
    store.remove_position("learning", "NIFTYBEES")
    assert store.positions("learning") == []
    assert len(store.positions("realistic")) == 1
    store.remove_position("learning", "NIFTYBEES")        # removing a missing row is a no-op


def test_cash_set_get_update(store):
    store.set_cash("realistic", 1000.0)
    store.set_cash("learning", 100000.0)
    store.set_cash("realistic", 157.25)
    assert store.cash("realistic") == 157.25
    assert store.cash("learning") == 100000.0


def test_cash_unseeded_book_raises_rather_than_returning_zero(store):
    assert not store.has_cash("realistic")
    with pytest.raises(KeyError):
        store.cash("realistic")
    store.set_cash("realistic", 0.0)
    assert store.has_cash("realistic")
    assert store.cash("realistic") == 0.0


# ---- state ----

def test_state_default_and_json_values(store):
    assert store.get_state("missing") is None
    assert store.get_state("missing", 7) == 7
    store.set_state("flag", False)
    store.set_state("obj", {"a": [1, 2], "b": None})
    store.set_state("nothing", None)
    assert store.get_state("flag", True) is False
    assert store.get_state("obj") == {"a": [1, 2], "b": None}
    assert store.get_state("nothing", "dflt") is None
    store.set_state("flag", True)
    assert store.get_state("flag") is True


# ---- ideas, cards, risk audit ----

def test_get_idea_missing_is_none(store):
    assert store.get_idea("nope") is None


def test_save_idea_overwrites_same_id(store):
    store.save_idea(idea())
    changed = idea()
    changed.stop = 265.0
    store.save_idea(changed)
    assert store.get_idea("i1").stop == 265.0


def test_cards_save_get_and_filter_by_status(store):
    assert store.get_card("c1") is None
    store.save_card("c1", {"card_id": "c1", "status": "open", "symbol": "NIFTYBEES"})
    store.save_card("c2", {"card_id": "c2", "status": "open", "symbol": "GOLDBEES"})
    store.save_card("c3", {"card_id": "c3", "status": "approved"})
    assert store.get_card("c1") == {"card_id": "c1", "status": "open", "symbol": "NIFTYBEES"}
    assert [c["card_id"] for c in store.cards_with_status("open")] == ["c1", "c2"]
    assert [c["card_id"] for c in store.cards_with_status("approved")] == ["c3"]
    assert store.cards_with_status("expired") == []


def test_card_status_change_moves_it_between_filters(store):
    store.save_card("c1", {"card_id": "c1", "status": "open"})
    store.save_card("c1", {"card_id": "c1", "status": "expired"})
    assert store.cards_with_status("open") == []
    assert store.get_card("c1")["status"] == "expired"


def test_risk_decision_approved_and_default(store):
    assert store.risk_approved("o1") is False          # nothing recorded
    store.save_risk_decision("o1", True, [], at(MON, "10:00"))
    store.save_risk_decision("o2", False, ["cost rule", "stale quote"], at(MON, "10:00"))
    assert store.risk_approved("o1") is True
    assert store.risk_approved("o2") is False


def test_risk_approved_follows_latest_decision(store):
    store.save_risk_decision("o1", True, [], at(MON, "10:00"))
    store.save_risk_decision("o1", False, ["entry band"], at(MON, "10:02"))
    assert store.risk_approved("o1") is False
    store.save_risk_decision("o1", True, [], at(MON, "10:03"))
    assert store.risk_approved("o1") is True


def test_risk_reasons_are_stored_as_json(tmp_path):
    s = Store(tmp_path / "t.db")
    s.save_risk_decision("o2", False, ["cost rule", "stale quote"], at(MON, "10:00"))
    s.close()
    row = raw_rows(tmp_path / "t.db", "SELECT approved, reasons FROM risk_log")[0]
    assert row[0] == 0 and json.loads(row[1]) == ["cost rule", "stale quote"]


# ---- agents and AI usage ----

def test_save_agent_output(tmp_path):
    s = Store(tmp_path / "t.db")
    s.save_agent_output("news", '{"items": []}', 0.12, at(MON, "08:30"))
    s.close()
    rows = raw_rows(tmp_path / "t.db", "SELECT agent, payload, cost_usd FROM agent_outputs")
    assert rows == [("news", '{"items": []}', 0.12)]


def test_ai_spend_month_sums(store):
    store.add_ai_usage("claude-opus-5-5", 1000, 200, 0, 0, 1.5, at(date(2026, 10, 5), "09:00"))
    store.add_ai_usage("claude-opus-5-5", 1000, 200, 0, 0, 2.0, at(date(2026, 10, 30), "15:00"))
    store.add_ai_usage("claude-opus-5-5", 1000, 200, 0, 0, 4.0, at(date(2026, 11, 1), "09:00"))
    assert store.ai_spend_usd(2026, 10) == pytest.approx(3.5)
    assert store.ai_spend_usd(2026, 11) == pytest.approx(4.0)
    assert store.ai_spend_usd(2026, 9) == 0.0


def test_ai_spend_month_boundaries_are_ist(store):
    # 19:00 UTC on 31 Oct is 00:30 IST on 1 Nov, so it belongs to November.
    store.add_ai_usage("m", 1, 1, 0, 0, 1.0, datetime(2026, 10, 31, 19, 0, tzinfo=timezone.utc))
    store.add_ai_usage("m", 1, 1, 0, 0, 2.0, datetime(2026, 10, 31, 18, 0, tzinfo=timezone.utc))
    assert store.ai_spend_usd(2026, 11) == pytest.approx(1.0)
    assert store.ai_spend_usd(2026, 10) == pytest.approx(2.0)


def test_ai_spend_december_rolls_into_next_year(store):
    store.add_ai_usage("m", 1, 1, 0, 0, 5.0, at(date(2026, 12, 31), "23:59"))
    store.add_ai_usage("m", 1, 1, 0, 0, 7.0, at(date(2027, 1, 1), "00:00"))
    assert store.ai_spend_usd(2026, 12) == pytest.approx(5.0)
    assert store.ai_spend_usd(2027, 1) == pytest.approx(7.0)


def test_ai_spend_on_day(store):
    store.add_ai_usage("m", 1, 1, 0, 0, 1.0, at(MON, "00:00"))
    store.add_ai_usage("m", 1, 1, 0, 0, 0.5, at(MON, "23:59"))
    store.add_ai_usage("m", 1, 1, 0, 0, 9.0, at(TUE, "00:00"))
    assert store.ai_spend_usd_on(MON) == pytest.approx(1.5)
    assert store.ai_spend_usd_on(TUE) == pytest.approx(9.0)
    assert store.ai_spend_usd_on(WED) == 0.0


# ---- journal and equity ----

def test_recent_lessons_newest_first_and_limited(store):
    store.add_lesson("first", at(MON, "16:00"))
    store.add_lesson("second", at(TUE, "16:00"))
    store.add_lesson("third", at(WED, "16:00"))
    assert store.recent_lessons(2) == ["third", "second"]
    assert store.recent_lessons(10) == ["third", "second", "first"]
    assert store.recent_lessons(0) == []


def test_equity_snapshot_roundtrip_and_overwrite(store):
    assert store.equity_snapshot("learning", MON) is None
    store.save_equity_snapshot("learning", MON, 100000.0)
    store.save_equity_snapshot("realistic", MON, 1000.0)
    store.save_equity_snapshot("learning", MON, 99500.5)
    assert store.equity_snapshot("learning", MON) == 99500.5
    assert store.equity_snapshot("realistic", MON) == 1000.0
    assert store.equity_snapshot("learning", TUE) is None


def test_equity_snapshot_rejects_datetime_instead_of_date(store):
    with pytest.raises(TypeError):
        store.save_equity_snapshot("learning", at(MON, "15:30"), 1.0)
    with pytest.raises(TypeError):
        store.equity_snapshot("learning", at(MON, "15:30"))


# ---- counters ----

def test_entries_on_ignores_exits(store):
    store.save_fill(fill(order_id="a", ts=at(MON, "10:00")))
    store.save_fill(fill(order_id="b", symbol="GOLDBEES", ts=at(MON, "11:00")))
    store.save_fill(fill(order_id="c", side=Side.SELL, is_exit=False, ts=at(MON, "12:00")))
    store.save_fill(fill(order_id="d", side=Side.SELL, is_exit=True, ts=at(MON, "13:00")))
    store.save_fill(fill(order_id="e", ts=at(TUE, "10:00")))
    store.save_fill(fill(order_id="f", book="realistic", ts=at(MON, "10:00")))
    assert store.entries_on("learning", MON) == 2
    assert store.entries_on("learning", TUE) == 1
    assert store.entries_on("learning", WED) == 0
    assert store.entries_on("realistic", MON) == 1


def test_entries_on_uses_ist_date_of_fill(store):
    # 18:45 UTC on 2 Nov is 00:15 IST on 3 Nov.
    store.save_fill(fill(ts=datetime(2026, 11, 2, 18, 45, tzinfo=timezone.utc)))
    assert store.entries_on("learning", MON) == 0
    assert store.entries_on("learning", TUE) == 1


def test_bought_today(store):
    store.save_fill(fill(symbol="NIFTYBEES", ts=at(MON, "10:00")))
    store.save_fill(fill(symbol="GOLDBEES", side=Side.SELL, ts=at(MON, "11:00")))
    assert store.bought_today("learning", "NIFTYBEES", MON) is True
    assert store.bought_today("learning", "NIFTYBEES", TUE) is False
    assert store.bought_today("learning", "GOLDBEES", MON) is False       # only sold
    assert store.bought_today("realistic", "NIFTYBEES", MON) is False


def test_sell_days_counts_distinct_non_exit_days(store):
    store.save_fill(fill(order_id="a", side=Side.SELL, ts=at(MON, "10:00")))
    store.save_fill(fill(order_id="a2", side=Side.SELL, ts=at(MON, "14:00")))      # same day
    store.save_fill(fill(order_id="b", side=Side.SELL, ts=at(TUE, "10:00")))
    store.save_fill(fill(order_id="c", side=Side.SELL, is_exit=True, ts=at(WED, "10:00")))
    assert store.sell_days_in_week("learning", WED) == 2


def test_sell_days_window_is_monday_through_d(store):
    sun = date(2026, 11, 1)
    next_mon = date(2026, 11, 9)
    store.save_fill(fill(order_id="a", side=Side.SELL, ts=at(sun, "10:00")))        # previous week
    store.save_fill(fill(order_id="b", side=Side.SELL, ts=at(MON, "10:00")))
    store.save_fill(fill(order_id="c", side=Side.SELL, ts=at(WED, "10:00")))        # after d
    store.save_fill(fill(order_id="d", side=Side.SELL, ts=at(next_mon, "10:00")))   # next week
    store.save_fill(fill(order_id="e", side=Side.BUY, ts=at(TUE, "10:00")))         # a buy
    store.save_fill(fill(order_id="f", book="realistic", side=Side.SELL, ts=at(TUE, "10:00")))
    assert store.sell_days_in_week("learning", TUE) == 1                            # Monday only
    assert store.sell_days_in_week("learning", WED) == 2
    assert store.sell_days_in_week("learning", sun) == 1                            # Mon 26 Oct..Sun 1 Nov
    assert store.sell_days_in_week("learning", MON) == 1
    assert store.sell_days_in_week("realistic", TUE) == 1
    assert store.sell_days_in_week("realistic", MON) == 0
