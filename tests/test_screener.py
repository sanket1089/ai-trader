"""Screener tests. Pure functions, no I/O."""

from aitrader.data.indicators import IndicatorSnapshot
from aitrader.data.screener import load_universe, screen

NIFTY_CSV = """Company Name,Industry,Symbol,Series,ISIN Code
Tata Consultancy Services Ltd.,Information Technology,TCS,EQ,INE467B01029
Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018
HDFC Bank Limited,Financial Services,HDFCBANK,EQ,INE040A01034
Mahindra & Mahindra Ltd.,Automobile and Auto Components,M&M,EQ,INE101A01026
Bajaj Auto Ltd.,Automobile and Auto Components,BAJAJ-AUTO,EQ,INE917I01010
State Bank of India,Financial Services,SBIN,EQ,INE062A01020
"""


def snap(symbol, close=100.0, pct=0.0, vol=1.0):
    return IndicatorSnapshot(
        symbol=symbol,
        close=close,
        sma20=close,
        sma50=close,
        rsi14=50.0,
        atr14=1.0,
        vol_ratio20=vol,
        pct_change_5d=pct,
        high_20d=close,
        low_20d=close,
    )


# --------------------------------------------------------------------------- load_universe
def test_load_universe_parses_csv():
    symbols, aliases = load_universe(NIFTY_CSV, etfs=["NIFTYBEES", "GOLDBEES"])

    assert symbols == ["TCS", "RELIANCE", "HDFCBANK", "M&M", "BAJAJ-AUTO", "SBIN", "NIFTYBEES", "GOLDBEES"]
    assert aliases["TCS"] == ["Tata Consultancy Services Ltd.", "Tata Consultancy Services"]
    assert aliases["HDFCBANK"] == ["HDFC Bank Limited", "HDFC Bank"]
    assert aliases["M&M"] == ["Mahindra & Mahindra Ltd.", "Mahindra & Mahindra"]
    assert aliases["SBIN"] == ["State Bank of India"]  # no suffix, so no duplicate short form
    assert aliases["NIFTYBEES"] == []
    assert set(aliases) == set(symbols)


def test_load_universe_strips_skips_blanks_and_dedupes():
    csv_text = (
        "﻿Company Name , Symbol \n"
        "  Infosys Ltd. ,  infy  \n"
        ",\n"
        "Blank Symbol Co,  \n"
        "   ,ORPHAN\n"
        "Infosys Ltd.,INFY\n"
    )

    symbols, aliases = load_universe(csv_text, etfs=[" niftybees ", "", "INFY"])

    assert symbols == ["INFY", "ORPHAN", "NIFTYBEES"]
    assert aliases["INFY"] == ["Infosys Ltd.", "Infosys"]
    assert aliases["ORPHAN"] == []  # no company name -> no aliases, but the symbol is still tracked
    assert aliases["NIFTYBEES"] == []


def test_load_universe_missing_columns_returns_etfs_only():
    symbols, aliases = load_universe("foo,bar\n1,2\n", etfs=["GOLDBEES"])
    assert symbols == ["GOLDBEES"]
    assert aliases == {"GOLDBEES": []}
    assert load_universe("", etfs=[]) == ([], {})


# --------------------------------------------------------------------------- screen
def test_screen_score_formula_orders_candidates():
    snaps = {
        "MOVER": snap("MOVER", pct=-4.0),  # 4
        "VOLUME": snap("VOLUME", vol=3.5),  # 2 * 2.5 = 5
        "NEWS": snap("NEWS", pct=0.5, vol=0.5),  # 0.5 + 0 (vol below 1 does not subtract) + 3 * 2 = 6.5
        "DULL": snap("DULL"),  # 0
    }

    out = screen(snaps, {"NEWS": 2}, held=set(), max_price=None, n=4)

    assert out == ["NEWS", "VOLUME", "MOVER", "DULL"]


def test_screen_always_includes_held():
    snaps = {
        "HELD_DULL": snap("HELD_DULL"),
        "A": snap("A", pct=9.0),
        "B": snap("B", pct=8.0),
        "C": snap("C", pct=7.0),
    }

    out = screen(snaps, {}, held={"HELD_DULL"}, max_price=None, n=2)

    assert out == ["HELD_DULL", "A"]  # held first, then top scorers up to n in total


def test_screen_held_exceeding_n_all_included_with_no_extras():
    snaps = {s: snap(s, pct=float(i)) for i, s in enumerate("ABCDE")}

    out = screen(snaps, {}, held={"A", "B", "C"}, max_price=None, n=2)

    assert out == ["C", "B", "A"]  # all held kept (best score first), no other symbols added


def test_screen_held_without_snapshot_still_included_and_counts_toward_n():
    snaps = {"A": snap("A", pct=5.0), "B": snap("B", pct=4.0)}

    out = screen(snaps, {}, held={"NOSNAP"}, max_price=None, n=2)

    assert out == ["NOSNAP", "A"]


def test_screen_respects_n_and_max_price():
    snaps = {
        "CHEAP1": snap("CHEAP1", close=50.0, pct=3.0),
        "CHEAP2": snap("CHEAP2", close=80.0, pct=2.0),
        "CHEAP3": snap("CHEAP3", close=100.0, pct=1.0),  # == max_price is allowed
        "PRICEY": snap("PRICEY", close=100.01, pct=50.0),  # excluded despite the best score
    }

    out = screen(snaps, {}, held=set(), max_price=100.0, n=2)
    assert out == ["CHEAP1", "CHEAP2"]

    out = screen(snaps, {}, held=set(), max_price=100.0, n=10)
    assert out == ["CHEAP1", "CHEAP2", "CHEAP3"]

    assert screen(snaps, {}, held=set(), max_price=None, n=1) == ["PRICEY"]


def test_screen_max_price_not_applied_to_held():
    snaps = {"PRICEY": snap("PRICEY", close=5000.0), "OK": snap("OK", close=50.0, pct=1.0)}

    out = screen(snaps, {}, held={"PRICEY"}, max_price=100.0, n=5)

    assert out == ["PRICEY", "OK"]


def test_screen_ties_broken_by_symbol_name():
    snaps = {s: snap(s, pct=2.0) for s in ["ZED", "ALPHA", "MID"]}

    assert screen(snaps, {}, held=set(), max_price=None, n=2) == ["ALPHA", "MID"]
    # held ties too
    assert screen(snaps, {}, held={"ZED", "ALPHA"}, max_price=None, n=3) == ["ALPHA", "ZED", "MID"]


def test_screen_n_zero_returns_only_held():
    snaps = {"A": snap("A", pct=5.0), "H": snap("H")}
    assert screen(snaps, {}, held={"H"}, max_price=None, n=0) == ["H"]
    assert screen(snaps, {}, held=set(), max_price=None, n=0) == []
