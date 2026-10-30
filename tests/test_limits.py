import logging
import math
from datetime import date, datetime, timedelta, timezone

import pytest

from aitrader.calendar import IST, at
from aitrader.charges import compute_charges
from aitrader.config import BookConfig, ChargesConfig
from aitrader.models import Position, Quote, Side
from aitrader.risk.equity import book_equity
from aitrader.risk.limits import LimitEval, LimitState, LossLimits
from aitrader.store import Store

CFG = ChargesConfig(brokerage_per_order=0.0, stt_pct=0.1, stamp_buy_pct=0.015,
                    exchange_pct=0.00322, sebi_pct=0.0001, gst_pct=18.0, dp_per_sell=15.34)

REALISTIC = BookConfig(name="realistic", capital=1000.0, daily_loss_limit=100.0, total_loss_limit=500.0,
                       max_entries_per_day=1, max_sell_days_per_week=3, max_position_pct=100.0)
LEARNING = BookConfig(name="learning", capital=100000.0, daily_loss_limit=10000.0,
                      total_loss_limit=50000.0, max_entries_per_day=3, max_sell_days_per_week=None,
                      max_position_pct=20.0)

BOOK = "realistic"
FRI = date(2026, 10, 30)
MON = date(2026, 11, 2)
TUE = MON + timedelta(days=1)
WED = MON + timedelta(days=2)
OPEN_MON = at(MON, "09:15")
NOON_MON = at(MON, "12:00")
NOON_TUE = at(TUE, "12:00")


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "t.db"


@pytest.fixture
def store(db_path):
    s = Store(db_path)
    s.set_cash(BOOK, 1000.0)
    yield s
    s.close()


@pytest.fixture
def limits(store):
    """Realistic book, Friday's close was 1000, no holidays."""
    store.save_equity_snapshot(BOOK, FRI, 1000.0)
    return LossLimits(store, REALISTIC, [])


def hold(store, symbol="NIFTYBEES", qty=3, avg_price=280.0, book=BOOK):
    store.upsert_position(Position(book=book, symbol=symbol, qty=qty, avg_price=avg_price, stop=270.0,
                                   target=300.0, opened_at=at(FRI, "11:00")))


def quote(price, symbol="NIFTYBEES"):
    return Quote(symbol=symbol, price=price, ts=NOON_MON)


# ---------------------------------------------------------------- book_equity


def test_equity_subtracts_estimated_sell_charges(store):
    store.set_cash(BOOK, 160.0)
    hold(store, qty=3, avg_price=275.0)
    eq = book_equity(store, BOOK, {"NIFTYBEES": quote(280.0)}, CFG)
    sell = compute_charges(Side.SELL, 3, 280.0, CFG).total
    assert sell > 15.34                      # the DP charge is part of it
    assert eq == pytest.approx(160.0 + 840.0 - sell)
    assert eq < 1000.0 - 15.34


def test_equity_with_no_holdings_is_cash(store):
    store.set_cash(BOOK, 987.65)
    assert book_equity(store, BOOK, {}, CFG) == 987.65


def test_equity_charges_each_holding_separately(store):
    store.set_cash(BOOK, 100.0)
    hold(store, "NIFTYBEES", qty=3, avg_price=280.0)
    hold(store, "GOLDBEES", qty=10, avg_price=70.0)
    quotes = {"NIFTYBEES": quote(280.0), "GOLDBEES": quote(72.0, "GOLDBEES")}
    expected = (100.0
                + 3 * 280.0 - compute_charges(Side.SELL, 3, 280.0, CFG).total
                + 10 * 72.0 - compute_charges(Side.SELL, 10, 72.0, CFG).total)
    assert book_equity(store, BOOK, quotes, CFG) == pytest.approx(expected)


def test_equity_uses_avg_price_and_warns_when_no_quote(store, caplog):
    store.set_cash(BOOK, 160.0)
    hold(store, qty=3, avg_price=275.0)
    with caplog.at_level(logging.WARNING, logger="aitrader.risk.equity"):
        eq = book_equity(store, BOOK, {}, CFG)
    assert eq == pytest.approx(160.0 + 3 * 275.0 - compute_charges(Side.SELL, 3, 275.0, CFG).total)
    assert any("NIFTYBEES" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("bad", [0.0, -5.0, math.nan, math.inf])
def test_equity_treats_an_unusable_quote_price_as_no_quote(store, caplog, bad):
    store.set_cash(BOOK, 160.0)
    hold(store, qty=3, avg_price=275.0)
    with caplog.at_level(logging.WARNING, logger="aitrader.risk.equity"):
        eq = book_equity(store, BOOK, {"NIFTYBEES": quote(bad)}, CFG)
    assert eq == pytest.approx(160.0 + 3 * 275.0 - compute_charges(Side.SELL, 3, 275.0, CFG).total)
    assert any("NIFTYBEES" in r.getMessage() for r in caplog.records)


def test_equity_ignores_quotes_for_other_symbols_and_other_books(store):
    store.set_cash(BOOK, 160.0)
    store.set_cash("learning", 5000.0)
    hold(store, qty=3, avg_price=275.0)
    hold(store, "GOLDBEES", qty=50, avg_price=70.0, book="learning")
    eq = book_equity(store, BOOK, {"GOLDBEES": quote(90.0, "GOLDBEES")}, CFG)
    assert eq == pytest.approx(160.0 + 3 * 275.0 - compute_charges(Side.SELL, 3, 275.0, CFG).total)


def test_equity_of_an_unseeded_book_fails_loudly(store):
    with pytest.raises(KeyError):
        book_equity(store, "nope", {}, CFG)


# ---------------------------------------------------------------- limits: basics


def test_baseline_defaults_to_capital_and_is_persisted(store):
    ll = LossLimits(store, REALISTIC, [])
    assert ll.baseline() == 1000.0
    assert store.get_state("baseline:realistic") == 1000.0
    assert ll.stop_id() == 0


def test_existing_baseline_is_not_overwritten_by_a_new_instance(store):
    store.set_state("baseline:realistic", 640.0)
    assert LossLimits(store, REALISTIC, []).baseline() == 640.0


def test_fresh_state_is_active(limits):
    assert limits.state(NOON_MON) is LimitState.ACTIVE
    ev = limits.evaluate(1000.0, NOON_MON)
    assert ev == LimitEval(LimitState.ACTIVE, False, 0.0, 0.0)


def test_loss_numbers_are_reported_signed(limits):
    ev = limits.evaluate(940.0, NOON_MON)
    assert ev.daily_loss == pytest.approx(60.0)
    assert ev.total_loss == pytest.approx(60.0)
    ev = limits.evaluate(1030.0, NOON_MON)
    assert ev.daily_loss == pytest.approx(-30.0)       # a gain is a negative loss
    assert ev.state is LimitState.ACTIVE


# ---------------------------------------------------------------- limits: daily


def test_daily_pause_at_exact_limit(limits):
    just_under = limits.evaluate(1000.0 - 99.99, NOON_MON)
    assert just_under.state is LimitState.ACTIVE
    assert just_under.newly_triggered is False
    exact = limits.evaluate(1000.0 - 100.0, NOON_MON)
    assert exact.state is LimitState.DAILY_PAUSED
    assert exact.newly_triggered is True
    assert exact.daily_loss == pytest.approx(100.0)


def test_daily_limit_tolerates_float_noise_from_unrounded_cash(limits):
    # a loss of "exactly 100" that arrives as 99.9999999 after float arithmetic must still pause
    noisy = 900.0 + 1e-7
    assert 99.99999 < 1000.0 - noisy < 100.0
    assert limits.evaluate(noisy, NOON_MON).state is LimitState.DAILY_PAUSED


def test_daily_limit_tolerance_is_tight(limits):
    assert limits.evaluate(1000.0 - 99.9999, NOON_MON).state is LimitState.ACTIVE


def test_daily_pause_persists_for_the_rest_of_the_day(limits):
    limits.evaluate(890.0, OPEN_MON)
    ev = limits.evaluate(1000.0, at(MON, "14:00"))      # recovered intraday: still paused
    assert ev.state is LimitState.DAILY_PAUSED
    assert limits.state(at(MON, "15:25")) is LimitState.DAILY_PAUSED


def test_daily_pause_clears_next_trading_day(limits, store):
    limits.evaluate(890.0, NOON_MON)
    assert limits.state(NOON_MON) is LimitState.DAILY_PAUSED
    assert limits.state(NOON_TUE) is LimitState.ACTIVE
    store.save_equity_snapshot(BOOK, MON, 890.0)        # Monday's close becomes Tuesday's reference
    ev = limits.evaluate(890.0, NOON_TUE)
    assert ev.state is LimitState.ACTIVE
    assert ev.newly_triggered is False
    assert ev.daily_loss == pytest.approx(0.0)


def test_daily_pause_clears_across_a_weekend(store):
    store.save_equity_snapshot(BOOK, FRI, 1000.0)
    ll = LossLimits(store, REALISTIC, [])
    ll.evaluate(850.0, at(FRI, "12:00"))
    assert ll.state(at(FRI, "15:00")) is LimitState.DAILY_PAUSED
    assert ll.state(at(date(2026, 10, 31), "12:00")) is LimitState.ACTIVE      # Saturday
    assert ll.state(NOON_MON) is LimitState.ACTIVE


def test_overnight_gap_counts_toward_daily(limits):
    # previous snapshot 1000, opening equity 880 (gap down on the held ETF) -> paused before any trade
    ev = limits.evaluate(880.0, OPEN_MON)
    assert ev.state is LimitState.DAILY_PAUSED
    assert ev.newly_triggered is True
    assert ev.daily_loss == pytest.approx(120.0)


def test_daily_reference_falls_back_to_baseline_when_no_snapshot(store):
    ll = LossLimits(store, REALISTIC, [])               # no snapshot saved at all
    ev = ll.evaluate(880.0, NOON_MON)
    assert ev.state is LimitState.DAILY_PAUSED
    assert ev.daily_loss == pytest.approx(120.0)


def test_daily_reference_is_the_previous_trading_day_skipping_holidays(store):
    holiday = date(2026, 11, 6)                          # Friday
    monday = date(2026, 11, 9)
    store.save_equity_snapshot(BOOK, date(2026, 11, 5), 1000.0)    # Thursday: the real reference
    store.save_equity_snapshot(BOOK, holiday, 850.0)               # never a trading day: must be ignored
    ll = LossLimits(store, REALISTIC, [holiday])
    ev = ll.evaluate(950.0, at(monday, "12:00"))
    assert ev.daily_loss == pytest.approx(50.0)
    assert ev.state is LimitState.ACTIVE


def test_the_days_own_snapshot_is_not_the_reference(limits, store):
    store.save_equity_snapshot(BOOK, MON, 500.0)         # today's (e.g. stale or early) snapshot
    assert limits.evaluate(950.0, NOON_MON).daily_loss == pytest.approx(50.0)


def test_now_is_converted_to_ist_for_the_trading_day(limits):
    # 20:00 UTC on Sunday is 01:30 IST on Monday 2026-11-02
    utc_sun = datetime(2026, 11, 1, 20, 0, tzinfo=timezone.utc)
    assert utc_sun.astimezone(IST).date() == MON
    ev = limits.evaluate(880.0, utc_sun)
    assert ev.state is LimitState.DAILY_PAUSED
    assert limits.state(NOON_MON) is LimitState.DAILY_PAUSED         # stored against IST Monday


# ---------------------------------------------------------------- limits: total / hard stop


def test_hard_stop_at_total_limit_and_persists(limits):
    ev = limits.evaluate(500.0, NOON_MON)
    assert ev.state is LimitState.HARD_STOPPED
    assert ev.newly_triggered is True
    assert ev.total_loss == pytest.approx(500.0)
    # equity recovers: still hard-stopped, same day and later days
    assert limits.evaluate(1000.0, at(MON, "14:00")).state is LimitState.HARD_STOPPED
    assert limits.state(NOON_TUE) is LimitState.HARD_STOPPED
    assert limits.evaluate(1200.0, at(date(2026, 11, 9), "10:00")).state is LimitState.HARD_STOPPED


def test_total_limit_boundary(limits, store):
    assert limits.evaluate(500.01, NOON_MON).state is not LimitState.HARD_STOPPED
    assert limits.evaluate(500.0 + 1e-7, NOON_MON).state is LimitState.HARD_STOPPED   # float noise


def test_hard_stop_is_measured_from_the_baseline_not_the_previous_day(store):
    store.save_equity_snapshot(BOOK, FRI, 560.0)         # a long slide: yesterday already at 560
    ll = LossLimits(store, REALISTIC, [])
    ev = ll.evaluate(499.0, NOON_MON)
    assert ev.daily_loss == pytest.approx(61.0)
    assert ev.total_loss == pytest.approx(501.0)
    assert ev.state is LimitState.HARD_STOPPED


def test_hard_stop_precedence_over_daily(limits):
    # one evaluation breaches both limits
    ev = limits.evaluate(450.0, NOON_MON)
    assert ev.state is LimitState.HARD_STOPPED
    assert ev.newly_triggered is True
    assert ev.daily_loss == pytest.approx(550.0)
    assert limits.state(NOON_MON) is LimitState.HARD_STOPPED


def test_hard_stop_wins_when_it_follows_a_daily_pause(limits):
    assert limits.evaluate(880.0, OPEN_MON).state is LimitState.DAILY_PAUSED
    ev = limits.evaluate(480.0, NOON_MON)
    assert ev.state is LimitState.HARD_STOPPED
    assert ev.newly_triggered is True                    # DAILY_PAUSED -> HARD_STOPPED is a transition


def test_a_daily_breach_while_hard_stopped_is_not_a_new_trigger(limits):
    limits.evaluate(480.0, OPEN_MON)
    ev = limits.evaluate(300.0, NOON_MON)
    assert ev.state is LimitState.HARD_STOPPED
    assert ev.newly_triggered is False


# ---------------------------------------------------------------- limits: newly_triggered


def test_newly_triggered_only_once(limits):
    first = limits.evaluate(890.0, NOON_MON)
    again = limits.evaluate(880.0, at(MON, "13:00"))
    third = limits.evaluate(1000.0, at(MON, "14:00"))
    assert (first.newly_triggered, again.newly_triggered, third.newly_triggered) == (True, False, False)

    hard = limits.evaluate(400.0, at(MON, "14:30"))
    assert hard.newly_triggered is True
    assert limits.evaluate(400.0, at(MON, "14:33")).newly_triggered is False


def test_a_daily_pause_can_trigger_again_on_a_later_day(limits, store):
    assert limits.evaluate(890.0, NOON_MON).newly_triggered is True
    store.save_equity_snapshot(BOOK, MON, 1000.0)        # recovered by the close
    assert limits.evaluate(1000.0, NOON_TUE).newly_triggered is False
    ev = limits.evaluate(890.0, at(TUE, "14:00"))
    assert ev.state is LimitState.DAILY_PAUSED
    assert ev.newly_triggered is True


def test_state_survives_a_restart_on_the_same_store(limits, store):
    assert limits.evaluate(890.0, NOON_MON).newly_triggered is True
    restarted = LossLimits(store, REALISTIC, [])
    assert restarted.state(NOON_MON) is LimitState.DAILY_PAUSED
    ev = restarted.evaluate(890.0, at(MON, "12:03"))
    assert ev.state is LimitState.DAILY_PAUSED
    assert ev.newly_triggered is False

    assert restarted.evaluate(300.0, at(MON, "12:06")).newly_triggered is True
    again = LossLimits(store, REALISTIC, [])
    assert again.state(NOON_TUE) is LimitState.HARD_STOPPED
    ev = again.evaluate(300.0, at(MON, "12:09"))
    assert ev.state is LimitState.HARD_STOPPED
    assert ev.newly_triggered is False


def test_state_survives_reopening_the_database_file(db_path):
    s1 = Store(db_path)
    s1.set_cash(BOOK, 1000.0)
    s1.save_equity_snapshot(BOOK, FRI, 1000.0)
    assert LossLimits(s1, REALISTIC, []).evaluate(450.0, NOON_MON).newly_triggered is True
    s1.close()

    s2 = Store(db_path)
    try:
        ll = LossLimits(s2, REALISTIC, [])
        assert ll.state(NOON_TUE) is LimitState.HARD_STOPPED
        ev = ll.evaluate(450.0, NOON_MON)
        assert ev.state is LimitState.HARD_STOPPED
        assert ev.newly_triggered is False
    finally:
        s2.close()


def test_state_needs_no_equity_and_writes_nothing(limits, store):
    keys = ("hard_stop:realistic", "paused_on:realistic")
    before = [store.get_state(k) for k in keys]
    assert limits.state(NOON_MON) is LimitState.ACTIVE
    assert [store.get_state(k) for k in keys] == before


# ---------------------------------------------------------------- limits: reset


def test_reset_baseline_clears_stop_and_bumps_stop_id(limits, store):
    limits.evaluate(480.0, NOON_MON)
    assert limits.state(NOON_MON) is LimitState.HARD_STOPPED
    assert limits.stop_id() == 0

    limits.reset_baseline(480.0)

    assert limits.baseline() == 480.0
    assert limits.stop_id() == 1
    assert limits.state(NOON_MON) is LimitState.ACTIVE
    assert not store.get_state("hard_stop:realistic")
    ev = limits.evaluate(480.0, at(MON, "12:03"))
    assert ev.total_loss == pytest.approx(0.0)
    assert ev.state is not LimitState.HARD_STOPPED

    limits.reset_baseline(470.0)
    assert limits.stop_id() == 2


def test_reset_baseline_clears_todays_daily_pause(limits):
    limits.evaluate(880.0, OPEN_MON)
    assert limits.state(NOON_MON) is LimitState.DAILY_PAUSED
    limits.reset_baseline(880.0)
    assert limits.state(NOON_MON) is LimitState.ACTIVE


def test_total_loss_is_measured_from_the_new_baseline_after_reset(limits):
    limits.evaluate(480.0, NOON_MON)
    limits.reset_baseline(900.0)
    assert limits.evaluate(850.0, NOON_TUE).total_loss == pytest.approx(50.0)
    ev = limits.evaluate(400.0, NOON_TUE)                        # 500 below the new baseline
    assert ev.total_loss == pytest.approx(500.0)
    assert ev.state is LimitState.HARD_STOPPED
    assert ev.newly_triggered is True                            # a second stop is a new trigger
    assert limits.stop_id() == 1                                 # evaluate never touches stop_id


def test_reset_baseline_survives_a_restart(limits, store):
    limits.evaluate(480.0, NOON_MON)
    limits.reset_baseline(480.0)
    restarted = LossLimits(store, REALISTIC, [])
    assert restarted.baseline() == 480.0
    assert restarted.stop_id() == 1
    assert restarted.state(NOON_MON) is LimitState.ACTIVE


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_reset_baseline_rejects_an_unusable_equity(limits, bad):
    with pytest.raises(ValueError):
        limits.reset_baseline(bad)
    assert limits.baseline() == 1000.0
    assert limits.stop_id() == 0


# ---------------------------------------------------------------- limits: guards and isolation


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_evaluate_rejects_non_finite_equity(limits, store, bad):
    # a NaN compares false against every limit, so it must never read as "no loss"
    with pytest.raises(ValueError):
        limits.evaluate(bad, NOON_MON)
    assert limits.state(NOON_MON) is LimitState.ACTIVE


def test_naive_datetimes_are_rejected(limits):
    with pytest.raises(ValueError):
        limits.evaluate(1000.0, datetime(2026, 11, 2, 12, 0))
    with pytest.raises(ValueError):
        limits.state(datetime(2026, 11, 2, 12, 0))


def test_books_are_independent(store):
    store.set_cash("learning", 100000.0)
    store.save_equity_snapshot(BOOK, FRI, 1000.0)
    store.save_equity_snapshot("learning", FRI, 100000.0)
    real = LossLimits(store, REALISTIC, [])
    learn = LossLimits(store, LEARNING, [])

    assert real.evaluate(880.0, NOON_MON).state is LimitState.DAILY_PAUSED
    assert learn.state(NOON_MON) is LimitState.ACTIVE
    assert learn.evaluate(95000.0, NOON_MON).state is LimitState.ACTIVE          # 5,000 < 10,000
    assert learn.evaluate(90000.0, NOON_MON).state is LimitState.DAILY_PAUSED    # exactly 10,000

    real.evaluate(400.0, NOON_MON)
    assert real.state(NOON_MON) is LimitState.HARD_STOPPED
    assert learn.state(NOON_MON) is LimitState.DAILY_PAUSED
    learn.reset_baseline(90000.0)
    assert real.stop_id() == 0
    assert real.state(NOON_MON) is LimitState.HARD_STOPPED


def test_holidays_iterable_is_copied_at_construction(store):
    holiday = date(2026, 11, 6)                          # Friday
    monday = date(2026, 11, 9)
    store.save_equity_snapshot(BOOK, date(2026, 11, 5), 1000.0)    # Thursday: the reference if Friday is a holiday
    store.save_equity_snapshot(BOOK, holiday, 850.0)               # the reference if Friday is NOT a holiday
    ll = LossLimits(store, REALISTIC, (d for d in [holiday]))      # a one-shot generator
    first = ll.evaluate(950.0, at(monday, "12:00"))
    second = ll.evaluate(950.0, at(monday, "12:05"))               # the generator would be exhausted by now
    assert first.daily_loss == pytest.approx(50.0)                 # 1000 - 950, not 850 - 950
    assert second.daily_loss == pytest.approx(50.0)
    assert ll.state(at(monday, "12:10")) is LimitState.ACTIVE


# ---------------------------------------------------------------- limits: daily reference fallback


def test_a_missing_previous_close_does_not_hide_a_real_daily_loss(store):
    # true prior close 1300 is missing, so the latest earlier snapshot (1300 on the Wednesday
    # before) is the reference; falling back to the 1000 baseline would read 1150 as a gain
    store.save_equity_snapshot(BOOK, date(2026, 10, 28), 1300.0)
    ll = LossLimits(store, REALISTIC, [])
    ev = ll.evaluate(1150.0, NOON_MON)
    assert ev.daily_loss == pytest.approx(150.0)
    assert ev.total_loss == pytest.approx(-150.0)
    assert ev.state is LimitState.DAILY_PAUSED
    assert ev.newly_triggered is True


def test_the_latest_earlier_snapshot_wins_over_older_ones(store):
    store.save_equity_snapshot(BOOK, date(2026, 10, 27), 1500.0)
    store.save_equity_snapshot(BOOK, date(2026, 10, 28), 1100.0)
    ll = LossLimits(store, REALISTIC, [])
    assert ll.evaluate(1050.0, NOON_MON).daily_loss == pytest.approx(50.0)


def test_the_previous_trading_days_snapshot_is_still_the_reference_when_it_exists(store):
    store.save_equity_snapshot(BOOK, date(2026, 10, 28), 1300.0)
    store.save_equity_snapshot(BOOK, FRI, 1000.0)
    ll = LossLimits(store, REALISTIC, [])
    ev = ll.evaluate(950.0, NOON_MON)
    assert ev.daily_loss == pytest.approx(50.0)
    assert ev.state is LimitState.ACTIVE


def test_the_baseline_is_the_reference_only_when_no_earlier_snapshot_exists(store):
    ll = LossLimits(store, REALISTIC, [])
    assert ll.evaluate(920.0, NOON_MON).daily_loss == pytest.approx(80.0)        # no snapshots at all


def test_a_same_day_or_later_snapshot_is_never_the_reference(store):
    store.save_equity_snapshot(BOOK, MON, 1300.0)
    store.save_equity_snapshot(BOOK, TUE, 1300.0)
    ll = LossLimits(store, REALISTIC, [])
    ev = ll.evaluate(920.0, NOON_MON)
    assert ev.daily_loss == pytest.approx(80.0)          # baseline 1000, not 1300
    assert ev.state is LimitState.ACTIVE


def test_a_missing_previous_close_is_logged(store, caplog):
    store.save_equity_snapshot(BOOK, date(2026, 10, 28), 1300.0)
    ll = LossLimits(store, REALISTIC, [])
    with caplog.at_level(logging.WARNING, logger="aitrader.risk.limits"):
        ll.evaluate(1300.0, NOON_MON)
    assert any("snapshot" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
