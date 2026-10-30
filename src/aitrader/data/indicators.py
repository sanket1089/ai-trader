"""Technical indicators from daily bars. Plain pandas, no TA library.

Contract: ``compute_indicators`` returns a snapshot whose every number is a finite ``float``,
or ``None`` when the history cannot support one. It never returns a snapshot containing NaN.
"""

import math
from dataclasses import dataclass

import pandas as pd

MIN_ROWS = 50
_COLUMNS = ["open", "high", "low", "close", "volume"]
_WILDER_ALPHA = 1 / 14


@dataclass(frozen=True)
class IndicatorSnapshot:
    symbol: str
    close: float
    sma20: float
    sma50: float
    rsi14: float
    atr14: float
    vol_ratio20: float
    pct_change_5d: float  # percent, close[-1] / close[-6] - 1
    high_20d: float  # over the last 20 rows including the latest
    low_20d: float


def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    """Rows with a complete OHLCV record and a positive close, as floats; ``None`` if unusable."""
    if df is None or df.empty or any(c not in df.columns for c in _COLUMNS):
        return None
    clean = df[_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    clean = clean.dropna()
    return clean[clean["close"] > 0]


def _wilder(series: pd.Series) -> float:
    return float(series.ewm(alpha=_WILDER_ALPHA, adjust=False).mean().iloc[-1])


def _rsi14(close: pd.Series) -> float:
    delta = close.diff().iloc[1:]  # the first diff is NaN
    avg_gain = _wilder(delta.clip(lower=0))
    avg_loss = _wilder(-delta.clip(upper=0))
    if avg_loss == 0:
        return 50.0 if avg_gain == 0 else 100.0  # flat series: neutral; only gains: maximal
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return _wilder(true_range.iloc[1:])  # the first row has no previous close


def compute_indicators(symbol: str, df: pd.DataFrame) -> IndicatorSnapshot | None:
    """Snapshot of ``df`` (columns ``open, high, low, close, volume``, oldest row first).

    Rows with NaN or a close <= 0 are discarded first; fewer than 50 valid rows gives ``None``.
    """
    bars = _clean(df)
    if bars is None or len(bars) < MIN_ROWS:
        return None

    close, high, low, volume = bars["close"], bars["high"], bars["low"], bars["volume"]

    prev_volume_mean = float(volume.iloc[-21:-1].mean())
    vol_ratio = float(volume.iloc[-1]) / prev_volume_mean if prev_volume_mean > 0 else 0.0

    snapshot = IndicatorSnapshot(
        symbol=symbol,
        close=float(close.iloc[-1]),
        sma20=float(close.iloc[-20:].mean()),
        sma50=float(close.iloc[-50:].mean()),
        rsi14=_rsi14(close),
        atr14=_atr14(high, low, close),
        vol_ratio20=vol_ratio,
        pct_change_5d=(float(close.iloc[-1]) / float(close.iloc[-6]) - 1.0) * 100.0,
        high_20d=float(high.iloc[-20:].max()),
        low_20d=float(low.iloc[-20:].min()),
    )
    numbers = [v for k, v in vars(snapshot).items() if k != "symbol"]
    if not all(math.isfinite(v) for v in numbers):
        return None
    return snapshot
