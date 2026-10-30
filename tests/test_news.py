"""News tests. Fully offline: ``parse`` is always a fake, feedparser never touches the network."""

import logging
import time
from datetime import datetime, timedelta, timezone

from aitrader.calendar import IST
from aitrader.data.news import NewsItem, fetch_news, prefilter

SINCE = datetime(2026, 10, 30, 9, 0, tzinfo=IST)  # 03:30 UTC


def utc_struct(y, mo, d, h, mi):
    """What feedparser puts in ``published_parsed``: a UTC struct_time."""
    return time.struct_time((y, mo, d, h, mi, 0, 0, 0, 0))


def entry(title, link, parsed=None, summary="", key="published_parsed"):
    e = {"title": title, "link": link, "summary": summary}
    if parsed is not None:
        e[key] = parsed
    return e


def feed(entries, title="Fake Feed", bozo=0, status=None):
    parsed = {"feed": {"title": title}, "entries": entries, "bozo": bozo}
    if status is not None:
        parsed["status"] = status  # absent on local parses and fakes; feedparser sets it for HTTP fetches
    return parsed


def fake_parse(by_url):
    def parse(url):
        result = by_url[url]
        if isinstance(result, Exception):
            raise result
        return result

    return parse


def item(title, summary="", link=None):
    return NewsItem(
        title=title,
        summary=summary,
        link=link or f"https://x/{title}",
        published=datetime(2026, 10, 30, 10, 0, tzinfo=IST),
        source="t",
    )


# --------------------------------------------------------------------------- fetch_news
def test_fetch_news_dedupes_and_filters_since():
    by_url = {
        "f1": feed(
            [
                entry("Fresh", "https://x/a", utc_struct(2026, 10, 30, 5, 0), summary="<p>Hello &amp; bye</p>"),
                entry("Old", "https://x/old", utc_struct(2026, 10, 30, 3, 0)),  # 08:30 IST < since
            ],
            title="Feed One",
        ),
        "f2": feed(
            [
                entry("Fresh again", "https://x/a", utc_struct(2026, 10, 30, 6, 0)),  # same link: dropped
                entry("Other", "https://x/b", utc_struct(2026, 10, 30, 7, 30)),
            ],
            title="Feed Two",
        ),
    }

    items = fetch_news(["f1", "f2"], SINCE, parse=fake_parse(by_url))

    assert [i.link for i in items] == ["https://x/b", "https://x/a"]  # newest first
    by_link = {i.link: i for i in items}
    a = by_link["https://x/a"]
    assert a.title == "Fresh"  # first occurrence wins
    assert a.source == "Feed One"
    assert a.summary == "Hello & bye"  # tags stripped, entities decoded
    assert a.published == datetime(2026, 10, 30, 10, 30, tzinfo=IST)  # 05:00 UTC
    assert a.published.utcoffset() == timedelta(hours=5, minutes=30)
    assert by_link["https://x/b"].source == "Feed Two"


def test_fetch_news_published_is_aware_ist_and_since_boundary_kept():
    boundary = utc_struct(2026, 10, 30, 3, 30)  # exactly 09:00 IST == since
    items = fetch_news(["f"], SINCE, parse=fake_parse({"f": feed([entry("T", "https://x/t", boundary)])}))
    assert len(items) == 1
    assert items[0].published.tzinfo is IST
    assert items[0].published == SINCE


def test_fetch_news_accepts_naive_since_as_ist():
    by_url = {"f": feed([entry("T", "https://x/t", utc_struct(2026, 10, 30, 3, 0))])}  # 08:30 IST
    assert fetch_news(["f"], datetime(2026, 10, 30, 8, 0), parse=fake_parse(by_url))
    assert not fetch_news(["f"], datetime(2026, 10, 30, 9, 0), parse=fake_parse(by_url))


def test_fetch_news_uses_updated_parsed_and_drops_undated(caplog):
    by_url = {
        "f": feed(
            [
                entry("Updated only", "https://x/u", utc_struct(2026, 10, 30, 6, 0), key="updated_parsed"),
                entry("No date", "https://x/nodate"),
                entry("Bad date", "https://x/bad", utc_struct(2026, 13, 45, 99, 0)),
            ]
        )
    }
    with caplog.at_level(logging.DEBUG):
        items = fetch_news(["f"], SINCE, parse=fake_parse(by_url))

    assert [i.link for i in items] == ["https://x/u"]
    assert "No date" in caplog.text  # dropped with a debug log, never guessed


def test_fetch_news_drops_entries_without_link_or_title():
    by_url = {
        "f": feed(
            [
                entry("", "https://x/notitle", utc_struct(2026, 10, 30, 5, 0)),
                entry("No link", "", utc_struct(2026, 10, 30, 5, 0)),
                entry("Fine", "https://x/fine", utc_struct(2026, 10, 30, 5, 0)),
            ]
        )
    }
    assert [i.link for i in fetch_news(["f"], SINCE, parse=fake_parse(by_url))] == ["https://x/fine"]


def test_failing_feed_skipped(caplog):
    by_url = {
        "bad": RuntimeError("boom"),
        "bozo": feed([], bozo=1),
        "good": feed([entry("Ok", "https://x/ok", utc_struct(2026, 10, 30, 5, 0))]),
    }
    with caplog.at_level(logging.WARNING):
        items = fetch_news(["bad", "bozo", "good"], SINCE, parse=fake_parse(by_url))

    assert [i.link for i in items] == ["https://x/ok"]
    assert "bad" in caplog.text
    assert "bozo" in caplog.text


def test_http_error_status_feed_skipped_with_warning(caplog):
    # feedparser reports an HTTP 5xx with an empty body as status=503, bozo=0, entries=[]: no exception
    by_url = {
        "https://feeds/down": feed([], status=503),
        "https://feeds/gone": feed([], status=404),
        "https://feeds/good": feed([entry("Ok", "https://x/ok", utc_struct(2026, 10, 30, 5, 0))], status=200),
    }
    with caplog.at_level(logging.DEBUG):
        items = fetch_news(list(by_url), SINCE, parse=fake_parse(by_url))

    assert [i.link for i in items] == ["https://x/ok"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert any("https://feeds/down" in r.getMessage() and "503" in r.getMessage() for r in warnings)
    assert any("https://feeds/gone" in r.getMessage() and "404" in r.getMessage() for r in warnings)
    assert not any("https://feeds/good" in r.getMessage() for r in warnings)


def test_empty_200_feed_is_legitimate_and_not_warned(caplog):
    by_url = {"https://feeds/quiet": feed([], status=200), "https://feeds/local": feed([])}  # no status key
    with caplog.at_level(logging.DEBUG):
        assert fetch_news(list(by_url), SINCE, parse=fake_parse(by_url)) == []

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "https://feeds/quiet" in caplog.text  # noted at debug level only


def test_bozo_feed_with_entries_is_kept():
    by_url = {"f": feed([entry("Ok", "https://x/ok", utc_struct(2026, 10, 30, 5, 0))], bozo=1)}
    assert len(fetch_news(["f"], SINCE, parse=fake_parse(by_url))) == 1


def test_fetch_news_source_falls_back_to_feed_url():
    by_url = {"https://feeds/x": {"feed": {}, "entries": [entry("T", "https://x/t", utc_struct(2026, 10, 30, 5, 0))]}}
    items = fetch_news(["https://feeds/x"], SINCE, parse=fake_parse(by_url))
    assert items[0].source == "https://feeds/x"


def test_text_keeps_angle_brackets_that_are_not_tags():
    by_url = {
        "f": feed(
            [
                entry(
                    "Sensex <80,000 but Nifty >25,000 holds",
                    "https://x/num",
                    utc_struct(2026, 10, 30, 5, 0),
                    summary="<b>bold</b> text",
                ),
                entry(
                    "Rupee <b>weakens</b> past 88",
                    "https://x/tag",
                    utc_struct(2026, 10, 30, 4, 0),
                    summary="<p>Line one</p><br/><!-- note -->two &lt;3",
                ),
            ]
        )
    }

    items = {i.link: i for i in fetch_news(["f"], SINCE, parse=fake_parse(by_url))}

    assert items["https://x/num"].title == "Sensex <80,000 but Nifty >25,000 holds"
    assert items["https://x/num"].summary == "bold text"
    assert items["https://x/tag"].title == "Rupee weakens past 88"
    assert items["https://x/tag"].summary == "Line one two <3"


# --------------------------------------------------------------------------- prefilter
def test_prefilter_matches_alias_case_insensitive():
    items = [
        item("tata consultancy services wins big deal"),
        item("Weather today is sunny"),
        item("Reliance shares rally", summary="oil to telecom"),
    ]
    symbols = {"TCS": ["Tata Consultancy Services", "Tata Consultancy Services Ltd."], "RELIANCE": ["Reliance Industries"]}

    kept = prefilter(items, symbols, market_keywords=[])

    assert [i.title for i in kept] == ["tata consultancy services wins big deal", "Reliance shares rally"]


def test_prefilter_word_boundary():
    items = [item("TCSL announces results"), item("TCS announces results")]

    kept = prefilter(items, {"TCS": []}, market_keywords=[])

    assert [i.title for i in kept] == ["TCS announces results"]


def test_prefilter_symbols_with_punctuation():
    items = [
        item("M&M posts record tractor sales"),
        item("Tractor sales rise at M&MX"),
        item("Mahindra and Mahindra news"),
        item("BAJAJ-AUTO exports jump"),
        item("Bajaj-Autos exports jump"),
        item("Sensex opens higher (BAJAJ-AUTO leads)"),
    ]
    symbols = {"M&M": [], "BAJAJ-AUTO": []}

    kept = prefilter(items, symbols, market_keywords=[])

    assert [i.title for i in kept] == [
        "M&M posts record tractor sales",
        "BAJAJ-AUTO exports jump",
        "Sensex opens higher (BAJAJ-AUTO leads)",
    ]


def test_prefilter_market_keywords_and_summary():
    items = [
        item("RBI holds repo rate steady"),
        item("Quiet local story", summary="Crude oil surges past $90"),
        item("Cricket final tonight"),
    ]

    kept = prefilter(items, {"TCS": []}, market_keywords=["RBI", "crude", "Sensex"])

    assert [i.title for i in kept] == ["RBI holds repo rate steady", "Quiet local story"]


def test_prefilter_ignores_blank_terms_and_empty_input():
    items = [item("Anything at all")]
    assert prefilter(items, {"": [""], "TCS": ["  "]}, market_keywords=["", "  "]) == []
    assert prefilter([], {"TCS": []}, market_keywords=["RBI"]) == []
