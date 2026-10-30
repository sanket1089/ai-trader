"""NSE end-of-day closes from the CM UDiFF bhavcopy.

These are the authoritative daily closes (used to mark P&L and measure loss limits). Every
failure, including holidays and files that are not yet published, yields ``{}`` so the caller
can fall back to the last yfinance daily close.
"""

import csv
import io
import logging
import math
import zipfile
from datetime import date

import httpx

from aitrader.data import BROWSER_USER_AGENT

log = logging.getLogger(__name__)

BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
)
# Equity series only: EQ (rolling settlement) and BE (trade-for-trade). Everything else
# (bonds, SME, ETF-like series, ...) is out of scope.
_SERIES = frozenset({"EQ", "BE"})
_REQUIRED_COLUMNS = ("TckrSymb", "SctySrs", "ClsPric")


def fetch_closes(d: date, http_get=httpx.get) -> dict[str, float]:
    """``{symbol: close}`` for EQ/BE rows of the bhavcopy of ``d``; ``{}`` on any failure."""
    ymd = d.strftime("%Y%m%d")
    try:
        resp = http_get(
            BHAVCOPY_URL.format(ymd=ymd),
            headers={"User-Agent": BROWSER_USER_AGENT, "Accept": "*/*"},
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return _parse(resp.content)
    except Exception as exc:  # noqa: BLE001 - any failure means "no bhavcopy", never a crash
        log.warning("bhavcopy for %s unavailable: %s: %s", ymd, type(exc).__name__, exc)
        return {}


def _parse(content: bytes) -> dict[str, float]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("zip contains no CSV")
        text = zf.read(names[0]).decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(text))
    reader.fieldnames = [f.strip() for f in reader.fieldnames or []]
    missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"CSV is missing columns {missing}")

    closes: dict[str, float] = {}
    for row in reader:
        if (row["SctySrs"] or "").strip() not in _SERIES:
            continue
        symbol = (row["TckrSymb"] or "").strip()
        try:
            price = float(row["ClsPric"])
        except (TypeError, ValueError):
            continue  # blank / malformed close
        if symbol and math.isfinite(price) and price > 0:
            closes[symbol] = price
    return closes
