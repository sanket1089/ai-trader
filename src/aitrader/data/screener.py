"""Universe loading and the deterministic shortlist handed to the analysts."""

import csv
import io
import re

from aitrader.data.indicators import IndicatorSnapshot

# " Ltd.", " Ltd", " Limited" at the end of a company name.
_SUFFIX_RE = re.compile(r"\s+(?:ltd\.?|limited)\s*$", re.IGNORECASE)


def _aliases(company: str) -> list[str]:
    """The company name and, when it differs, the form without a trailing ``Ltd.``/``Limited``."""
    if not company:
        return []
    out = [company]
    short = _SUFFIX_RE.sub("", company).strip()
    if short and short != company:
        out.append(short)
    return out


def load_universe(csv_text: str, etfs: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Symbols and aliases from the NSE Nifty 100 CSV (columns ``Symbol``, ``Company Name``).

    Header and cell whitespace is stripped and rows without a symbol are skipped. ``etfs`` are
    appended (when not already present) with no aliases. Returns ``(symbols, aliases)`` where
    ``aliases`` has an entry, possibly empty, for every symbol.
    """
    symbols: list[str] = []
    aliases: dict[str, list[str]] = {}

    reader = csv.DictReader(io.StringIO(csv_text.lstrip("﻿")))
    for row in reader:
        cells = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items() if k is not None}
        symbol = cells.get("symbol", "").upper()
        if not symbol or symbol in aliases:
            continue
        symbols.append(symbol)
        aliases[symbol] = _aliases(cells.get("company name", ""))

    for etf in etfs:
        symbol = etf.strip().upper()
        if symbol and symbol not in aliases:
            symbols.append(symbol)
            aliases[symbol] = []
    return symbols, aliases


def _score(snap: IndicatorSnapshot | None, news_hits: int) -> float:
    if snap is None:
        return 3.0 * news_hits
    return abs(snap.pct_change_5d) + 2.0 * max(snap.vol_ratio20 - 1.0, 0.0) + 3.0 * news_hits


def screen(
    snaps: dict[str, IndicatorSnapshot],
    news_hits: dict[str, int],
    held: set[str],
    max_price: float | None,
    n: int,
) -> list[str]:
    """Shortlist of symbols to analyse: held first, then the best-scoring others.

    Score = |pct_change_5d| + 2 * max(vol_ratio20 - 1, 0) + 3 * news_hits. Held symbols are
    always included (even without a snapshot, and even when that exceeds ``n``) and count toward
    ``n``; the remaining slots go to the top scorers priced at or below ``max_price`` (``None``
    means no cap; it never applies to held symbols). Ties break on symbol name, so the result is
    deterministic.
    """

    def rank(symbol: str) -> tuple[float, str]:
        return (-_score(snaps.get(symbol), news_hits.get(symbol, 0)), symbol)

    held_sorted = sorted(held, key=rank)
    slots = n - len(held_sorted)
    if slots <= 0:
        return held_sorted

    candidates = [
        s
        for s, snap in snaps.items()
        if s not in held and (max_price is None or snap.close <= max_price)
    ]
    return held_sorted + sorted(candidates, key=rank)[:slots]
