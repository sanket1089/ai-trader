"""RSS news fetch and the cheap code-level pre-filter that runs before any AI call.

Failure policy: a feed that raises, cannot be parsed or answers with an HTTP error status is
skipped with a warning; one bad feed never costs the others. An item without a parseable date is dropped (debug log), never dated
"now" by guesswork, so it can't slip past the ``since`` cutoff.
"""

import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser
import httpx

from aitrader.calendar import IST
from aitrader.data import BROWSER_USER_AGENT

log = logging.getLogger(__name__)

# Only tag-shaped things: "<" followed by a letter, "/" + letter, or "!" (comments, doctype).
# A bare "<" or ">" in a headline ("Sensex <80,000 but Nifty >25,000") is real text.
_TAG_RE = re.compile(r"</?[A-Za-z!][^>]*>")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NewsItem:
    title: str
    summary: str
    link: str
    published: datetime  # timezone-aware, IST
    source: str


def _text(value) -> str:
    """Plain text of an RSS field: tags stripped, entities decoded, whitespace collapsed."""
    if not value:
        return ""
    return _SPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", str(value)))).strip()


def _published(entry) -> datetime | None:
    """feedparser gives ``published_parsed`` / ``updated_parsed`` as UTC ``struct_time``."""
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if not parsed:
            continue
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc).astimezone(IST)
        except (ValueError, TypeError, OverflowError):
            continue
    return None


def _items_from_feed(url: str, parsed) -> list[NewsItem]:
    source = _text((parsed.get("feed") or {}).get("title")) or url
    items: list[NewsItem] = []
    for entry in parsed.get("entries") or []:
        title = _text(entry.get("title"))
        link = str(entry.get("link") or "").strip()
        if not title or not link:
            log.debug("dropping entry without title or link from %s", url)
            continue
        published = _published(entry)
        if published is None:
            log.debug("dropping undated news item %r from %s", title, url)
            continue
        items.append(
            NewsItem(
                title=title,
                summary=_text(entry.get("summary")),
                link=link,
                published=published,
                source=source,
            )
        )
    return items


def fetch_news(feeds: list[str], since: datetime, parse=feedparser.parse) -> list[NewsItem]:
    """Items from all ``feeds`` published at or after ``since``, deduped by link, newest first.

    A naive ``since`` is read as IST. The first occurrence of a link wins.
    """
    if since.tzinfo is None:
        since = since.replace(tzinfo=IST)

    seen: set[str] = set()
    out: list[NewsItem] = []
    for url in feeds:
        try:
            parsed = parse(url)
        except Exception as exc:
            log.warning("news feed %s failed: %s", url, exc)
            continue
        # feedparser does not raise on HTTP errors: a 5xx with an empty body comes back as
        # status=503, bozo=0, entries=[]. ``status`` is absent for local parses and fakes.
        status = parsed.get("status")
        if isinstance(status, int) and status >= 400:
            log.warning("news feed %s returned HTTP %d; skipping", url, status)
            continue
        if parsed.get("bozo") and not parsed.get("entries"):
            log.warning("news feed %s could not be parsed: %s", url, parsed.get("bozo_exception"))
            continue
        if not parsed.get("entries"):
            log.debug("news feed %s has no entries", url)  # a quiet feed is legitimate
        for item in _items_from_feed(url, parsed):
            if item.published < since or item.link in seen:
                continue
            seen.add(item.link)
            out.append(item)
    out.sort(key=lambda i: i.published, reverse=True)  # stable: ties keep feed order
    return out


def _term_pattern(terms: list[str]) -> re.Pattern | None:
    """One case-insensitive regex matching any term as a whole token.

    ``\\b`` is wrong for terms like ``M&M`` or ``BAJAJ-AUTO`` (it fires inside them), so the
    edges are lookarounds on alphanumerics instead: ``TCS`` matches in "TCS rallies" and
    "(TCS)" but not in "TCSL".
    """
    cleaned = sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True)
    if not cleaned:
        return None
    body = "|".join(re.escape(t) for t in cleaned)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", re.IGNORECASE)


def prefilter(
    items: list[NewsItem],
    symbols: dict[str, list[str]],
    market_keywords: list[str],
) -> list[NewsItem]:
    """Items whose title or summary mentions a symbol, an alias or a market keyword.

    ``symbols`` maps each symbol to its aliases (e.g. the company name). Input order is kept.
    """
    terms = list(market_keywords)
    for symbol, aliases in symbols.items():
        terms.append(symbol)
        terms.extend(aliases)
    pattern = _term_pattern(terms)
    if pattern is None:
        return []
    return [i for i in items if pattern.search(f"{i.title}\n{i.summary}")]


def count_hits(items: list[NewsItem], symbols: dict[str, list[str]]) -> dict[str, int]:
    """Number of items mentioning each symbol (or one of its aliases), with the prefilter's matcher.

    Symbols with no hits are left out.
    """
    hits: dict[str, int] = {}
    texts = [f"{i.title}\n{i.summary}" for i in items]
    for symbol, aliases in symbols.items():
        pattern = _term_pattern([symbol, *aliases])
        if pattern is None:
            continue
        n = sum(1 for text in texts if pattern.search(text))
        if n:
            hits[symbol] = n
    return hits


def http_parse(url: str, http_get=httpx.get, timeout: float = 15):
    """``feedparser.parse`` over a timeout-bounded HTTP fetch (feedparser's own fetch can hang).

    An HTTP error status or any exception gives a failed-feed result (``bozo`` true, no entries,
    the status when there is one), which ``fetch_news`` warns about and skips. Only the error's
    type is kept: its message may hold the URL.
    """
    try:
        resp = http_get(url, timeout=timeout, headers={"User-Agent": BROWSER_USER_AGENT},
                        follow_redirects=True)
    except Exception as exc:  # noqa: BLE001 - a dead feed is skipped, never fatal
        return {"bozo": True, "bozo_exception": type(exc).__name__, "entries": []}
    status = getattr(resp, "status_code", None)
    if not isinstance(status, int) or status >= 400:
        return {"bozo": True, "bozo_exception": f"HTTP {status}", "entries": [], "status": status}
    return feedparser.parse(resp.content)
