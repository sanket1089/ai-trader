"""Indicator tests. Pure pandas, no I/O."""

import math
from dataclasses import fields

import pandas as pd
import pytest

from aitrader.data.indicators import IndicatorSnapshot, compute_indicators


def frame(closes, volumes=None, highs=None, lows=None):
    n = len(closes)
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs if highs is not None else closes,
            "low": lows if lows is not None else closes,
            "close": closes,
            "volume": volumes if volumes is not None else [1000] * n,
        },
        index=pd.date_range("2026-01-01", periods=n, freq="B"),
    )


def test_sma_on_linear_series():
    snap = compute_indicators("X", frame([float(i) for i in range(1, 61)]))

    assert snap is not None
    assert snap.symbol == "X"
    assert snap.close == 60.0
    assert snap.sma20 == 50.5  # mean(41..60)
    assert snap.sma50 == 35.5  # mean(11..60)


def test_rsi_monotonic_up_is_100():
    snap = compute_indicators("X", frame([float(i) for i in range(1, 61)]))
    assert snap.rsi14 == 100.0


def test_rsi_monotonic_down_is_0_and_flat_is_50():
    down = compute_indicators("X", frame([float(i) for i in range(60, 0, -1)]))
    flat = compute_indicators("X", frame([100.0] * 60))
    assert down.rsi14 == 0.0
    assert flat.rsi14 == 50.0


def test_rsi_wilder_matches_reference():
    closes = [100 + 3 * math.sin(i / 3) + i * 0.1 for i in range(80)]
    s = pd.Series(closes)
    d = s.diff().iloc[1:]
    ag = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    al = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    expected = 100 - 100 / (1 + ag / al)

    snap = compute_indicators("X", frame(closes))

    assert snap.rsi14 == pytest.approx(expected)
    assert 0 < snap.rsi14 < 100


def test_atr_constant_true_range():
    # closes rise 1/day, high = close + 1, low = close - 1, so prev close = low:
    # TR = max(h - l = 2, |h - pc| = 2, |l - pc| = 0) = 2 on every row, and a Wilder average of 2s is 2
    closes = [100.0 + i for i in range(60)]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]

    snap = compute_indicators("X", frame(closes, highs=highs, lows=lows))

    assert snap.atr14 == pytest.approx(2.0)


def test_atr_reference_with_gap():
    closes = [100.0] * 59 + [110.0]
    highs = [101.0] * 59 + [111.0]
    lows = [99.0] * 59 + [109.0]
    prev = pd.Series(closes).shift(1)
    h, l = pd.Series(highs), pd.Series(lows)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1).iloc[1:]
    expected = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]

    snap = compute_indicators("X", frame(closes, highs=highs, lows=lows))

    assert snap.atr14 == pytest.approx(expected)
    assert snap.atr14 > 2.0  # the gap day (TR = 11) lifts it above the flat-day TR of 2


def test_vol_ratio_uses_previous_20_excluding_last():
    volumes = [1000] * 59 + [3000]
    snap = compute_indicators("X", frame([50.0] * 60, volumes=volumes))
    assert snap.vol_ratio20 == pytest.approx(3.0)

    # the spike is the last bar: an older spike outside the 20-bar window must not count
    volumes = [9000] + [1000] * 38 + [2000] * 20 + [4000]
    snap = compute_indicators("X", frame([50.0] * 60, volumes=volumes))
    assert snap.vol_ratio20 == pytest.approx(2.0)


def test_vol_ratio_zero_when_previous_mean_is_zero():
    volumes = [0] * 59 + [5000]
    snap = compute_indicators("X", frame([50.0] * 60, volumes=volumes))
    assert snap.vol_ratio20 == 0.0


def test_pct_change_5d_and_20d_range():
    closes = [100.0] * 54 + [100.0, 101.0, 102.0, 103.0, 104.0, 110.0]
    highs = [c + 2 for c in closes]
    lows = [c - 3 for c in closes]

    snap = compute_indicators("X", frame(closes, highs=highs, lows=lows))

    assert snap.pct_change_5d == pytest.approx((110.0 / 100.0 - 1) * 100)  # close[-1] / close[-6]
    assert snap.high_20d == 112.0
    assert snap.low_20d == 97.0

    # window is the last 20 rows including the last: a high 20 bars back is excluded
    highs2 = [c + 2 for c in closes]
    highs2[-20] = 400.0  # exactly the first row of the window
    highs2[-21] = 900.0  # just outside the window
    snap = compute_indicators("X", frame(closes, highs=highs2, lows=lows))
    assert snap.high_20d == 400.0


def test_short_history_returns_none():
    assert compute_indicators("X", frame([float(i) for i in range(1, 50)])) is None  # 49 rows
    assert compute_indicators("X", frame([float(i) for i in range(1, 51)])) is not None  # 50 rows
    assert compute_indicators("X", pd.DataFrame()) is None
    assert compute_indicators("X", pd.DataFrame({"close": [1.0] * 60})) is None  # missing columns


def test_zero_or_negative_closes_are_dropped_before_counting():
    closes = [float(i) for i in range(1, 51)]
    # 50 valid rows plus two zero-close rows interleaved: the zeros must not leak into any value
    df = frame(closes[:25] + [0.0] + closes[25:] + [0.0])
    snap = compute_indicators("X", df)
    assert snap is not None
    assert snap.close == 50.0
    assert snap.sma50 == pytest.approx(25.5)

    # 49 valid rows + 1 zero row: only 49 valid remain
    assert compute_indicators("X", frame(closes[:49] + [0.0])) is None


def test_nan_rows_are_dropped_and_result_has_no_nan():
    closes = [float(i) for i in range(1, 61)]
    df = frame(closes)
    df.loc[df.index[10], "volume"] = float("nan")
    snap = compute_indicators("X", df)
    assert snap is not None
    for f in fields(IndicatorSnapshot):
        v = getattr(snap, f.name)
        if f.name != "symbol":
            assert isinstance(v, float) and math.isfinite(v), f.name

    # too many NaN rows -> not enough history
    df2 = frame(closes)
    df2.loc[df2.index[:15], "close"] = float("nan")
    assert compute_indicators("X", df2) is None


def test_snapshot_fields_are_floats():
    snap = compute_indicators("X", frame([float(i) for i in range(1, 61)], volumes=list(range(1000, 1060))))
    for name in ("close", "sma20", "sma50", "rsi14", "atr14", "vol_ratio20", "pct_change_5d", "high_20d", "low_20d"):
        assert type(getattr(snap, name)) is float, name
