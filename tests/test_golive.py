"""Go-live report (spec §9) over a real Store."""

from datetime import date, timedelta

import pytest
from pipeline_fakes import LEARN, REAL, SETTINGS

from aitrader.calendar import at
from aitrader.golive import GoLiveReport, evaluate_go_live, format_report
from aitrader.models import Fill, Order, OrderStatus, Side
from aitrader.store import Store

NOW = at(date(2027, 1, 9), "10:00")          # a Saturday


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "g.db")
    yield s
    s.close()


def passing(store, start=NOW - timedelta(weeks=9)):
    store.set_state("phase1_start", start.date().isoformat())
    for i in range(15):
        ts = start + timedelta(days=i + 1)
        store.save_fill(Fill(f"s{i}", LEARN, "ITC", Side.SELL, 1, 400.0, 16.0, i % 2 == 0, ts))
    for i in range(3):
        order = Order(id=f"o{i}", book=LEARN, symbol="ITC", side=Side.BUY, qty=1, limit_price=400.0,
                      reason="r", is_exit=False, idea_id=None, created_at=start + timedelta(days=1),
                      status=OrderStatus.FILLED)
        store.save_order(order)
        store.save_risk_decision(order.id, True, [], start + timedelta(days=1))
    store.save_equity_snapshot(LEARN, NOW.date() - timedelta(days=1), 110000.0)    # +10%
    store.save_equity_snapshot(REAL, NOW.date() - timedelta(days=1), 1050.0)
    for key in ("tested_approval", "tested_stop", "tested_expiry"):
        store.set_state(key, True)
    store.add_ai_usage("claude-opus-5-5", 0, 0, 0, 0, 60.0, start + timedelta(days=2))


def failing_names(report: GoLiveReport) -> list[str]:
    return [name for name, ok, _ in report.checks if not ok]


def test_golive_ready_when_all_pass(store):
    passing(store)
    report = evaluate_go_live(store, SETTINGS, NOW, bench_return_pct=5.0)
    assert failing_names(report) == []
    assert report.ready is True
    assert len(report.checks) == 7
    assert "READY" in format_report(report)


def test_golive_not_ready_before_8_weeks(store):
    passing(store, start=NOW - timedelta(weeks=7))
    report = evaluate_go_live(store, SETTINGS, NOW, bench_return_pct=5.0)
    assert report.ready is False
    assert len(failing_names(report)) == 1 and "8 weeks" in failing_names(report)[0]


def test_golive_each_criterion_can_fail(store):
    passing(store)
    store.set_state("hard_stop:realistic", True)               # realistic book was stopped
    store.set_state("tested_expiry", False)
    store.save_order(Order(id="bypass", book=LEARN, symbol="ITC", side=Side.BUY, qty=1, limit_price=400.0,
                           reason="r", is_exit=False, idea_id=None, created_at=NOW - timedelta(days=3),
                           status=OrderStatus.FILLED))         # filled with no risk approval
    store.add_ai_usage("claude-opus-5-5", 0, 0, 0, 0, 120.0, NOW - timedelta(days=40))   # a month over cap
    report = evaluate_go_live(store, SETTINGS, NOW, bench_return_pct=12.0)            # bench beats learning
    assert len(failing_names(report)) == 5
    assert report.ready is False


def test_golive_missing_start_and_unknown_benchmark_fail(store):
    report = evaluate_go_live(store, SETTINGS, NOW, bench_return_pct=float("nan"))
    assert report.ready is False
    assert "8 weeks" in failing_names(report)[0]
