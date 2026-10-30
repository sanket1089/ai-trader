from datetime import date, datetime, timedelta, timezone

import pytest

from aitrader.calendar import IST, at, in_window, is_trading_day, now_ist, previous_trading_day


def test_weekend_not_trading():
    assert not is_trading_day(date(2026, 10, 31), [])          # Saturday
    assert not is_trading_day(date(2026, 11, 1), [])           # Sunday


def test_weekday_is_trading():
    assert is_trading_day(date(2026, 11, 2), [])               # Monday


def test_holiday_not_trading():
    assert not is_trading_day(date(2026, 11, 9), [date(2026, 11, 9)])


def test_previous_trading_day_skips_weekend():
    assert previous_trading_day(date(2026, 11, 2), []) == date(2026, 10, 30)


def test_previous_trading_day_skips_holiday_and_weekend():
    # Monday 9 Nov is a holiday, so the day before Tuesday 10 Nov is Friday 6 Nov.
    assert previous_trading_day(date(2026, 11, 10), [date(2026, 11, 9)]) == date(2026, 11, 6)


def test_previous_trading_day_accepts_one_shot_iterable():
    holidays = (d for d in [date(2026, 11, 9)])
    assert previous_trading_day(date(2026, 11, 10), holidays) == date(2026, 11, 6)


def test_now_ist_is_aware():
    assert now_ist().utcoffset() == timedelta(hours=5, minutes=30)


def test_at_is_aware_ist():
    t = at(date(2026, 11, 2), "09:20")
    assert t.utcoffset() == timedelta(hours=5, minutes=30)
    assert (t.year, t.month, t.day, t.hour, t.minute, t.second) == (2026, 11, 2, 9, 20, 0)
    assert t.tzinfo == IST


@pytest.mark.parametrize("bad", ["9:20", "0920", "24:00", "09:60", "", "09:20:00"])
def test_at_rejects_malformed_hhmm(bad):
    with pytest.raises(ValueError):
        at(date(2026, 11, 2), bad)


def test_in_window_bounds():
    assert in_window(at(date(2026, 11, 2), "09:20"), "09:20", "15:10")
    assert not in_window(at(date(2026, 11, 2), "15:11"), "09:20", "15:10")


def test_in_window_end_is_inclusive_and_start_is_exclusive_before():
    assert in_window(at(date(2026, 11, 2), "15:10"), "09:20", "15:10")
    assert not in_window(at(date(2026, 11, 2), "09:19"), "09:20", "15:10")


def test_in_window_converts_other_timezones_to_ist():
    # 04:00 UTC is 09:30 IST, inside the entry window.
    t = datetime(2026, 11, 2, 4, 0, tzinfo=timezone.utc)
    assert in_window(t, "09:20", "15:10")


def test_in_window_rejects_naive_datetime():
    with pytest.raises(ValueError):
        in_window(datetime(2026, 11, 2, 10, 0), "09:20", "15:10")
