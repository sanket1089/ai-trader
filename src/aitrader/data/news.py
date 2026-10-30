"""RSS news fetch and the cheap code-level pre-filter that runs before any AI call.

Failure policy: a feed that raises or cannot be parsed is skipped with a warning; one bad feed
never costs the others. An item without a parseable date is dropped (debug log), never dated
"now" by guesswork, so it can't slip past the ``since`` cutoff.
"""

import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser

from aitrader.calendar import IST

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]*>")
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
        if parsed.get("bozo") and not parsed.get("entries"):
            log.warning("news feed %s could not be parsed: %s", url, parsed.get("bozo_exception"))
            continue
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
