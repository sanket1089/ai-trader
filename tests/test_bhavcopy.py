"""Bhavcopy tests. Fully offline: ``http_get`` is always a fake."""

import io
import logging
import zipfile
from datetime import date

from aitrader.data.bhavcopy import fetch_closes

CSV = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,LastPric
2026-10-30,2026-10-30,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,1400.00,1420.00,1390.00,1412.35,1412.35
2026-10-30,2026-10-30,CM,NSE,STK,1234,INE000B01011, SMALLCO ,BE,10.00,10.50,9.90,10.25,10.25
2026-10-30,2026-10-30,CM,NSE,STK,5678,INE000C01012,NCDBOND,N1,1000.00,1000.00,1000.00,1000.00,1000.00
2026-10-30,2026-10-30,CM,NSE,STK,9999,INE000D01013,BADPRICE,EQ,1.00,1.00,1.00,,1.00
2026-10-30,2026-10-30,CM,NSE,STK,8888,INE000E01014,ZEROPX,EQ,0.00,0.00,0.00,0.00,0.00
"""


def make_zip(csv_text: str = CSV, name: str = "BhavCopy_NSE_CM_0_0_0_20261030_F_0000.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, csv_text)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_bhavcopy_parses_eq_rows():
    calls = []

    def http_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(make_zip())

    closes = fetch_closes(date(2026, 10, 30), http_get=http_get)

    assert closes == {"RELIANCE": 1412.35, "SMALLCO": 10.25}  # EQ and BE kept; N1, empty and 0 prices dropped
    url, kwargs = calls[0]
    assert url == (
        "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261030_F_0000.csv.zip"
    )
    assert "Mozilla" in kwargs["headers"]["User-Agent"]


def test_bhavcopy_failure_returns_empty(caplog):
    def http_get(url, **kwargs):
        raise ConnectionError("network down")

    with caplog.at_level(logging.WARNING):
        assert fetch_closes(date(2026, 10, 30), http_get=http_get) == {}
    assert "20261030" in caplog.text


def test_bhavcopy_http_error_returns_empty():
    # NSE answers 404 for holidays / files not yet published.
    assert fetch_closes(date(2026, 10, 31), http_get=lambda url, **kw: FakeResponse(b"", status=404)) == {}


def test_bhavcopy_bad_payloads_return_empty():
    payloads = [
        b"not a zip at all",
        make_zip("A,B\n1,2\n"),  # CSV without the expected columns
        make_zip(""),  # empty CSV
    ]
    for payload in payloads:
        assert fetch_closes(date(2026, 10, 30), http_get=lambda url, p=payload, **kw: FakeResponse(p)) == {}
