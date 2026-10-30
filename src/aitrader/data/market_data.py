"""Quotes, history, fundamentals and index context from yfinance (free, delayed ~15 min).

Failure policy: this layer never raises for data problems and never invents a price. A symbol
whose latest price is missing, NaN or <= 0 is simply absent from the result (with a warning),
so the risk gate sees "no quote" rather than a bogus number.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import httpx
import pandas as pd
import yfinance

from aitrader.calendar import IST
from aitrader.config import RiskConfig
from aitrader.data import BROWSER_USER_AGENT
from aitrader.models import Quote

log = logging.getLogger(__name__)

HISTORY_COLUMNS = ["open", "high", "low", "close", "volume"]
FUNDAMENTAL_KEYS = (
    "trailingPE",
    "priceToBook",
    "returnOnEquity",
    "debtToEquity",
    "marketCap",
    "earningsGrowth",
    "sector",
)
FIIDII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"

# Index symbols used across the service -> Yahoo tickers.
_INDEX_TICKERS = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}


class MarketData(Protocol):
    def quote(self, symbol: str) -> Quote | None: ...

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Missing symbols are omitted."""
        ...

    def daily_history(self, symbol: str, days: int) -> pd.DataFrame:
        """Columns ``open, high, low, close, volume``; empty on failure."""
        ...

    def fundamentals(self, symbol: str) -> dict:
        """``{}`` on failure."""
        ...

    def index_snapshot(self) -> dict:
        """Keys ``nifty_pct, banknifty_pct, fii_net_cr, dii_net_cr``; values may be ``None``."""
        ...


def is_stale(q: Quote, now: datetime, cfg: RiskConfig) -> bool:
    """True when the quote is older than the feed delay plus the allowed extra minutes."""
    limit = timedelta(minutes=cfg.data_delay_minutes + cfg.stale_extra_minutes)
    return now - q.ts > limit


def _yahoo_ticker(symbol: str) -> str:
    s = symbol.strip().upper()
    if s in _INDEX_TICKERS:
        return _INDEX_TICKERS[s]
    if s.startswith("^") or s.endswith(".NS"):
        return s
    return f"{s}.NS"


def _close_series(frame: Any, ticker: str, allow_flat: bool) -> pd.Series | None:
    """The ``Close`` column of ``ticker`` from a download result, or ``None``.

    yfinance returns MultiIndex columns (ticker first with ``group_by="ticker"``); older
    versions returned flat columns for a single ticker, which is accepted when ``allow_flat``.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    cols = frame.columns
    if isinstance(cols, pd.MultiIndex):
        for level in (0, 1):
            if ticker in cols.get_level_values(level):
                sub = frame.xs(ticker, axis=1, level=level)
                break
        else:
            return None
    elif allow_flat:
        sub = frame
    else:
        return None
    if "Close" not in sub.columns:
        return None
    close = sub["Close"]
    return close if isinstance(close, pd.Series) else None


def _to_ist(ts: Any) -> datetime:
    """Aware IST datetime from a pandas timestamp; naive input is treated as UTC."""
    dt = pd.Timestamp(ts).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


def _clean_closes(close: pd.Series | None) -> pd.Series:
    if close is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(close, errors="coerce").dropna().sort_index()


def _num(value: Any) -> float | None:
    """Float from an NSE string such as ``"-1,234.50"``; ``None`` if not a finite number."""
    try:
        out = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _history_period(days: int) -> str:
    """Smallest valid yfinance period that safely covers ``days`` trading days."""
    for limit, period in ((3, "5d"), (15, "1mo"), (55, "3mo"), (110, "6mo"), (230, "1y"), (480, "2y")):
        if days <= limit:
            return period
    return "5y"


class YFinanceMarketData:
    def __init__(self, yf_module: Any = yfinance, http_get: Any = httpx.get):
        self._yf = yf_module
        self._http_get = http_get

    # -- quotes -----------------------------------------------------------------------
    def quote(self, symbol: str) -> Quote | None:
        return self.quotes([symbol]).get(symbol)

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        wanted = {s: _yahoo_ticker(s) for s in dict.fromkeys(symbols)}  # de-duplicated, ordered
        if not wanted:
            return {}
        tickers = list(dict.fromkeys(wanted.values()))
        try:
            frame = self._yf.download(
                tickers, period="1d", interval="5m", group_by="ticker", progress=False
            )
        except Exception as exc:  # noqa: BLE001 - network/parse failures mean "no quotes"
            log.warning("quote download failed for %s: %s: %s", tickers, type(exc).__name__, exc)
            return {}

        out: dict[str, Quote] = {}
        for symbol, ticker in wanted.items():
            q = self._last_quote(symbol, _close_series(frame, ticker, allow_flat=len(tickers) == 1))
            if q is not None:
                out[symbol] = q
        return out

    @staticmethod
    def _last_quote(symbol: str, close: pd.Series | None) -> Quote | None:
        # The last bar this ticker actually printed: in a batched download a ticker is NaN on
        # timestamps where it had no trade. The timestamp is kept, so staleness still shows.
        closes = _clean_closes(close)
        if closes.empty:
            log.warning("no usable quote for %s (empty or NaN data); skipping", symbol)
            return None
        if not isinstance(closes.index, pd.DatetimeIndex):
            log.warning("quote for %s has no timestamps; skipping", symbol)
            return None
        price = float(closes.iloc[-1])
        if not math.isfinite(price) or price <= 0:
            log.warning("quote for %s has invalid price %r; skipping", symbol, price)
            return None
        return Quote(symbol, price, _to_ist(closes.index[-1]))

    # -- history / fundamentals --------------------------------------------------------
    def daily_history(self, symbol: str, days: int) -> pd.DataFrame:
        try:
            raw = self._yf.Ticker(_yahoo_ticker(symbol)).history(
                period=_history_period(days), interval="1d"
            )
            df = raw.rename(columns=str.lower)[HISTORY_COLUMNS].copy()
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            return df.dropna(subset=["close"]).tail(days)
        except Exception as exc:  # noqa: BLE001
            log.warning("daily history unavailable for %s: %s: %s", symbol, type(exc).__name__, exc)
            return pd.DataFrame(columns=HISTORY_COLUMNS)

    def fundamentals(self, symbol: str) -> dict:
        try:
            info = self._yf.Ticker(_yahoo_ticker(symbol)).info
            result = {k: info.get(k) for k in FUNDAMENTAL_KEYS}
        except Exception as exc:  # noqa: BLE001
            log.warning("fundamentals unavailable for %s: %s: %s", symbol, type(exc).__name__, exc)
            return {}
        if all(v is None for v in result.values()):
            # yfinance answers unknown tickers (and blocked/401 requests) with a stub dict.
            log.warning("fundamentals for %s came back empty", symbol)
            return {}
        return result

    # -- market context ----------------------------------------------------------------
    def index_snapshot(self) -> dict:
        nifty, banknifty = self._index_pcts(["^NSEI", "^NSEBANK"])
        fii, dii = self._fii_dii()
        return {"nifty_pct": nifty, "banknifty_pct": banknifty, "fii_net_cr": fii, "dii_net_cr": dii}

    def _index_pcts(self, tickers: list[str]) -> list[float | None]:
        """Percent change of the latest daily close over the one before it."""
        try:
            frame = self._yf.download(
                tickers, period="5d", interval="1d", group_by="ticker", progress=False
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("index download failed: %s: %s", type(exc).__name__, exc)
            return [None] * len(tickers)
        out: list[float | None] = []
        for ticker in tickers:
            closes = _clean_closes(_close_series(frame, ticker, allow_flat=len(tickers) == 1))
            if len(closes) < 2 or closes.iloc[-2] <= 0:
                log.warning("index %s: fewer than two usable daily closes", ticker)
                out.append(None)
                continue
            pct = (float(closes.iloc[-1]) / float(closes.iloc[-2]) - 1.0) * 100.0
            out.append(pct if math.isfinite(pct) else None)
        return out

    def _fii_dii(self) -> tuple[float | None, float | None]:
        """Net FII / DII cash-market flow in INR crore; ``(None, None)`` on any failure."""
        try:
            resp = self._http_get(
                FIIDII_URL,
                headers={
                    "User-Agent": BROWSER_USER_AGENT,
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.nseindia.com/reports/fii-dii",
                },
                timeout=10,
                follow_redirects=True,
            )
            resp.raise_for_status()
            rows = resp.json()
            if not isinstance(rows, list):
                raise ValueError(f"unexpected payload type {type(rows).__name__}")
            fii = dii = None
            for row in rows:
                category = str(row.get("category", "")).strip().upper()
                if category.startswith("FII"):
                    fii = _num(row.get("netValue"))
                elif category.startswith("DII"):
                    dii = _num(row.get("netValue"))
            return fii, dii
        except Exception as exc:  # noqa: BLE001 - NSE blocks bots often; this input is optional
            log.warning("FII/DII data unavailable: %s: %s", type(exc).__name__, exc)
            return None, None
