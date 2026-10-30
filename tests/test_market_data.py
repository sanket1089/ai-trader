"""Market data tests. Fully offline: ``yf_module`` and ``http_get`` are always fakes."""

import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest

from aitrader.calendar import IST
from aitrader.config import RiskConfig
from aitrader.data.market_data import YFinanceMarketData, is_stale
from aitrader.models import Quote

IST_OFFSET = timedelta(hours=5, minutes=30)


# --------------------------------------------------------------------------- fakes
def bars(closes, start="2026-10-30 09:30", tz="UTC", freq="5min"):
    """OHLCV frame with Title-case columns, like yfinance."""
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz=tz)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=idx,
    )


def grouped(**frames):
    """Multi-ticker frame as ``download(..., group_by="ticker")`` returns it."""
    return pd.concat(frames, axis=1)


class FakeTicker:
    def __init__(self, symbol, info=None, history=None, error=None):
        self.symbol = symbol
        self._info = info
        self._history = history
        self._error = error

    @property
    def info(self):
        if self._error:
            raise self._error
        return self._info

    def history(self, **kwargs):
        if self._error:
            raise self._error
        self.history_kwargs = kwargs
        return self._history


class FakeYF:
    """Stub for the ``yfinance`` module: ``download`` and ``Ticker`` only."""

    def __init__(self, download=None, ticker=None):
        self._download = download
        self._ticker = ticker
        self.download_calls = []
        self.ticker_calls = []

    def download(self, tickers, **kwargs):
        self.download_calls.append((tickers, kwargs))
        result = self._download(tickers, **kwargs) if callable(self._download) else self._download
        if isinstance(result, Exception):
            raise result
        return result

    def Ticker(self, symbol):  # noqa: N802 - mirrors yfinance's API
        self.ticker_calls.append(symbol)
        return self._ticker(symbol) if callable(self._ticker) else self._ticker


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def make_http_get(payload=None, error=None):
    calls = []

    def http_get(url, **kwargs):
        calls.append((url, kwargs))
        if error:
            raise error
        return FakeResponse(payload)

    http_get.calls = calls
    return http_get


def no_http(*_a, **_k):
    raise AssertionError("no HTTP expected in this test")


def risk_cfg(delay=15, extra=10) -> RiskConfig:
    return RiskConfig(
        entry_band_pct=0.5,
        exit_band_pct=1.0,
        cost_multiple=3.0,
        data_delay_minutes=delay,
        stale_extra_minutes=extra,
        entry_start="09:20",
        entry_end="15:10",
        exit_start="09:15",
        max_orders_per_sec=1,
        exit_check_minutes=3,
        exit_reprice_minutes=2,
        exit_max_reprices=3,
    )


# --------------------------------------------------------------------------- quotes
def test_quote_maps_ns_suffix_and_ist_ts():
    # 09:40 UTC is 15:10 IST.
    frame = grouped(**{"NIFTYBEES.NS": bars([250.0, 251.5], start="2026-10-30 09:35")})
    yf = FakeYF(download=frame)
    md = YFinanceMarketData(yf_module=yf, http_get=no_http)

    q = md.quote("NIFTYBEES")

    assert q == Quote("NIFTYBEES", 251.5, datetime(2026, 10, 30, 15, 10, tzinfo=IST))
    assert q.ts.tzinfo is not None and q.ts.utcoffset() == IST_OFFSET
    tickers, kwargs = yf.download_calls[0]
    assert list(tickers) == ["NIFTYBEES.NS"]
    assert kwargs == {"period": "1d", "interval": "5m", "group_by": "ticker", "progress": False}


def test_quote_ts_converted_when_index_already_ist():
    frame = grouped(**{"TCS.NS": bars([4000.0], start="2026-10-30 15:10", tz="Asia/Kolkata")})
    q = YFinanceMarketData(yf_module=FakeYF(download=frame), http_get=no_http).quote("TCS")
    assert q.ts == datetime(2026, 10, 30, 15, 10, tzinfo=IST)
    assert q.ts.utcoffset() == IST_OFFSET


def test_quote_naive_index_is_treated_as_utc():
    frame = grouped(**{"TCS.NS": bars([4000.0], start="2026-10-30 09:40", tz=None)})
    q = YFinanceMarketData(yf_module=FakeYF(download=frame), http_get=no_http).quote("TCS")
    assert q.ts == datetime(2026, 10, 30, 15, 10, tzinfo=IST)
    assert q.ts.utcoffset() == IST_OFFSET


def test_quote_accepts_flat_columns_for_a_single_ticker():
    md = YFinanceMarketData(yf_module=FakeYF(download=bars([99.0, 100.0])), http_get=no_http)
    assert md.quote("INFY").price == 100.0


def test_quote_empty_frame_returns_none(caplog):
    md = YFinanceMarketData(yf_module=FakeYF(download=pd.DataFrame()), http_get=no_http)
    with caplog.at_level(logging.WARNING):
        assert md.quote("TCS") is None
    assert "TCS" in caplog.text


def test_quote_download_returning_none_or_raising_gives_none():
    for result in (None, RuntimeError("boom")):
        md = YFinanceMarketData(yf_module=FakeYF(download=result), http_get=no_http)
        assert md.quote("TCS") is None
        assert md.quotes(["TCS", "INFY"]) == {}


@pytest.mark.parametrize("bad", [float("nan"), None, 0.0, -5.0])
def test_quote_nan_returns_none(bad, caplog):
    frame = grouped(**{"TCS.NS": bars([bad])})
    md = YFinanceMarketData(yf_module=FakeYF(download=frame), http_get=no_http)
    with caplog.at_level(logging.WARNING):
        assert md.quote("TCS") is None
    assert "TCS" in caplog.text


def test_quote_zero_last_price_does_not_fall_back_to_older_bar():
    frame = grouped(**{"TCS.NS": bars([100.0, 0.0])})
    assert YFinanceMarketData(yf_module=FakeYF(download=frame), http_get=no_http).quote("TCS") is None


def test_quote_skips_trailing_nan_bar_and_uses_last_real_bar():
    # In a multi-ticker download a ticker has NaN where it printed no bar; its quote is its last real bar.
    frame = grouped(**{"TCS.NS": bars([100.0, float("nan")], start="2026-10-30 09:30")})
    q = YFinanceMarketData(yf_module=FakeYF(download=frame), http_get=no_http).quote("TCS")
    assert q.price == 100.0
    assert q.ts == datetime(2026, 10, 30, 15, 0, tzinfo=IST)  # first bar: 09:30 UTC


def test_quotes_omits_missing_symbol():
    frame = grouped(
        **{
            "TCS.NS": bars([100.0, 101.0]),
            "DEAD.NS": bars([float("nan"), float("nan")]),  # yfinance returns NaN columns for failures
        }
    )
    yf = FakeYF(download=frame)
    md = YFinanceMarketData(yf_module=yf, http_get=no_http)

    result = md.quotes(["TCS", "DEAD", "ABSENT"])

    assert set(result) == {"TCS"}
    assert result["TCS"].price == 101.0
    assert len(yf.download_calls) == 1  # one batched download
    assert list(yf.download_calls[0][0]) == ["TCS.NS", "DEAD.NS", "ABSENT.NS"]


def test_quotes_empty_input_makes_no_call():
    yf = FakeYF(download=pd.DataFrame())
    assert YFinanceMarketData(yf_module=yf, http_get=no_http).quotes([]) == {}
    assert yf.download_calls == []


def test_index_symbols_map_to_yahoo_tickers():
    frame = grouped(**{"^NSEI": bars([25000.0]), "^NSEBANK": bars([55000.0])})
    yf = FakeYF(download=frame)
    md = YFinanceMarketData(yf_module=yf, http_get=no_http)

    result = md.quotes(["NIFTY", "BANKNIFTY"])

    assert list(yf.download_calls[0][0]) == ["^NSEI", "^NSEBANK"]
    assert result["NIFTY"].price == 25000.0
    assert result["BANKNIFTY"].price == 55000.0


# --------------------------------------------------------------------------- history / fundamentals
def test_daily_history_columns_and_length():
    raw = bars(list(range(1, 31)), start="2026-09-01", tz="Asia/Kolkata", freq="1D")
    raw["Dividends"] = 0.0
    raw["Stock Splits"] = 0.0
    yf = FakeYF(ticker=lambda s: FakeTicker(s, history=raw))
    md = YFinanceMarketData(yf_module=yf, http_get=no_http)

    df = md.daily_history("TCS", 20)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 20
    assert df["close"].iloc[-1] == 30
    assert yf.ticker_calls == ["TCS.NS"]


def test_daily_history_failure_returns_empty_frame_with_columns():
    yf = FakeYF(ticker=lambda s: FakeTicker(s, error=RuntimeError("boom")))
    df = YFinanceMarketData(yf_module=yf, http_get=no_http).daily_history("TCS", 20)
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_fundamentals_returns_selected_keys():
    info = {
        "trailingPE": 25.1,
        "priceToBook": 4.2,
        "returnOnEquity": 0.45,
        "debtToEquity": 8.0,
        "marketCap": 1_400_000_000_000,
        "earningsGrowth": 0.1,
        "sector": "Technology",
        "irrelevant": "dropped",
    }
    yf = FakeYF(ticker=lambda s: FakeTicker(s, info=info))
    f = YFinanceMarketData(yf_module=yf, http_get=no_http).fundamentals("TCS")
    assert set(f) == {
        "trailingPE", "priceToBook", "returnOnEquity", "debtToEquity",
        "marketCap", "earningsGrowth", "sector",
    }  # fmt: skip
    assert f["trailingPE"] == 25.1 and f["sector"] == "Technology"


def test_fundamentals_failure_gives_empty_dict(caplog):
    caplog.set_level(logging.WARNING)
    for ticker in (
        lambda s: FakeTicker(s, error=RuntimeError("boom")),
        lambda s: FakeTicker(s, info={}),
        lambda s: FakeTicker(s, info={"trailingPegRatio": None}),  # yfinance's answer for unknown tickers
    ):
        caplog.clear()
        md = YFinanceMarketData(yf_module=FakeYF(ticker=ticker), http_get=no_http)
        assert md.fundamentals("TCS") == {}
        assert "TCS" in caplog.text  # every failure mode logs a warning


# --------------------------------------------------------------------------- index snapshot
def _index_yf():
    def download(tickers, **kwargs):
        closes = {"^NSEI": [25000.0, 25250.0], "^NSEBANK": [50000.0, 49500.0]}
        t = [tickers] if isinstance(tickers, str) else list(tickers)
        return grouped(**{x: bars(closes[x], start="2026-10-29", tz="Asia/Kolkata", freq="1D") for x in t})

    return FakeYF(download=download)


FIIDII_PAYLOAD = [
    {"category": "DII **", "date": "30-Oct-2026", "buyValue": "11,000.50", "sellValue": "9,000.00", "netValue": "2,000.50"},
    {"category": "FII/FPI *", "date": "30-Oct-2026", "buyValue": "8,000.00", "sellValue": "9,234.50", "netValue": "-1,234.50"},
]  # fmt: skip


def test_index_snapshot_parses_pct_and_fii_dii():
    http = make_http_get(FIIDII_PAYLOAD)
    snap = YFinanceMarketData(yf_module=_index_yf(), http_get=http).index_snapshot()

    assert snap["nifty_pct"] == pytest.approx(1.0)
    assert snap["banknifty_pct"] == pytest.approx(-1.0)
    assert snap["fii_net_cr"] == pytest.approx(-1234.5)
    assert snap["dii_net_cr"] == pytest.approx(2000.5)
    url, kwargs = http.calls[0]
    assert url == "https://www.nseindia.com/api/fiidiiTradeReact"
    assert "Mozilla" in kwargs["headers"]["User-Agent"]


def test_index_snapshot_fii_failure_gives_none():
    http = make_http_get(error=RuntimeError("blocked"))
    snap = YFinanceMarketData(yf_module=_index_yf(), http_get=http).index_snapshot()

    assert snap["fii_net_cr"] is None
    assert snap["dii_net_cr"] is None
    assert snap["nifty_pct"] == pytest.approx(1.0)  # one failing source does not blank the others
    assert snap["banknifty_pct"] == pytest.approx(-1.0)


def test_index_snapshot_bad_fii_payload_gives_none():
    for payload in ({"unexpected": "shape"}, [], [{"category": "FII/FPI *", "netValue": "n/a"}], None):
        snap = YFinanceMarketData(yf_module=_index_yf(), http_get=make_http_get(payload)).index_snapshot()
        assert snap["fii_net_cr"] is None and snap["dii_net_cr"] is None


def test_index_snapshot_total_failure_has_all_keys_none():
    yf = FakeYF(download=RuntimeError("down"))
    md = YFinanceMarketData(yf_module=yf, http_get=make_http_get(error=RuntimeError("blocked")))
    assert md.index_snapshot() == {
        "nifty_pct": None, "banknifty_pct": None, "fii_net_cr": None, "dii_net_cr": None,
    }  # fmt: skip


# --------------------------------------------------------------------------- staleness
def test_is_stale_boundary():
    cfg = risk_cfg(delay=15, extra=10)
    now = datetime(2026, 10, 30, 12, 0, tzinfo=IST)
    fresh = Quote("TCS", 100.0, now - timedelta(minutes=25))
    stale = Quote("TCS", 100.0, now - timedelta(minutes=25, seconds=1))
    assert is_stale(fresh, now, cfg) is False
    assert is_stale(stale, now, cfg) is True


def test_is_stale_uses_configured_thresholds_and_tolerates_clock_skew():
    now = datetime(2026, 10, 30, 12, 0, tzinfo=IST)
    q = Quote("TCS", 100.0, now - timedelta(minutes=11))
    assert is_stale(q, now, risk_cfg(delay=5, extra=5)) is True
    assert is_stale(q, now, risk_cfg(delay=5, extra=6)) is False
    assert is_stale(Quote("TCS", 100.0, now + timedelta(minutes=3)), now, risk_cfg()) is False
