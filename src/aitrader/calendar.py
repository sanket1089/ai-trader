"""IST clock, NSE trading days and time windows.

All times are timezone-aware in ``ZoneInfo("Asia/Kolkata")``. The machine's local
time is never used.
"""

import re
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

_HHMM = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")


def now_ist() -> datetime:
    return datetime.now(IST)


def is_trading_day(d: date, holidays: Iterable[date]) -> bool:
    """Monday to Friday, and not an exchange holiday."""
    return d.weekday() < 5 and d not in frozenset(holidays)


def previous_trading_day(d: date, holidays: Iterable[date]) -> date:
    """The latest trading day strictly before ``d``."""
    holiday_set = frozenset(holidays)
    prev = d - timedelta(days=1)
    while not is_trading_day(prev, holiday_set):
        prev -= timedelta(days=1)
    return prev


def at(d: date, hhmm: str) -> datetime:
    """``d`` at the quoted 24-hour ``"HH:MM"`` wall-clock time, as an aware IST datetime."""
    m = _HHMM.fullmatch(hhmm)
    if m is None:
        raise ValueError(f'expected a 24-hour "HH:MM" time, got {hhmm!r}')
    return datetime.combine(d, time(int(m.group(1)), int(m.group(2))), tzinfo=IST)


def in_window(t: datetime, start: str, end: str) -> bool:
    """True if ``t`` falls between ``start`` and ``end`` (both inclusive) on its IST date."""
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("in_window needs a timezone-aware datetime")
    local = t.astimezone(IST)
    return at(local.date(), start) <= local <= at(local.date(), end)
