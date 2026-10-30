"""Live dry run: the morning of a trading day on REAL data with the REAL Claude API, placing no orders.

    .venv\\Scripts\\python scripts\\simulate_day.py [--config config.yaml]

It runs ``premarket`` and then ``open_session`` once (``dry_run=True``) and prints what the owner would
have received on Telegram: the news digest, the shortlist, the portfolio manager's ideas with their
reasons, the approval cards, and the AI cost of the run. Nothing is sent to Telegram (a ``PrintNotifier``
prints instead) and no order of any kind is placed. It uses its own database, ``data/simulate.db``,
which is deleted and recreated on every run, so the real ``data/aitrader.db`` is never touched.

It costs real money (about US$1-3 of Claude API calls), so it needs ``ANTHROPIC_API_KEY`` in ``.env``
next to ``config.yaml``. Without it the script says so and exits with code 2, before anything is built.
The key is never printed. Cards appear only when the quotes are fresh: run it on a trading day between
about 09:30 and 15:10 IST (before 09:40 the free feed is still catching up with the open).
"""

import argparse
import asyncio
import logging
import os
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

try:
    import aitrader  # noqa: F401
except ImportError:                      # not installed (pip install -e .): use the source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aitrader import pipeline
from aitrader.agents.schemas import NewsDigest
from aitrader.app import build_services
from aitrader.calendar import in_window, is_trading_day, now_ist
from aitrader.config import Settings, load_settings
from aitrader.execution import describe_error
from aitrader.pipeline_support import inr, load_model, usd

REPO_ROOT = Path(__file__).resolve().parents[1]
NO_KEY_MESSAGE = "Set ANTHROPIC_API_KEY in .env to run the live dry run"
SIMULATE_DB_NAME = "simulate.db"

# The script never talks to Telegram, but ``Settings`` requires both Telegram values. These placeholders
# stand in for them, and the real values (if any) are deliberately not read, so they can never end up in
# this script's memory or output.
_DUMMY_TELEGRAM_ENV = {"TELEGRAM_OWNER_CHAT_ID": "1", "TELEGRAM_BOT_TOKEN": "not-used-by-simulate-day"}


@dataclass
class PrintNotifier:
    """Stands in for the Telegram bot: prints each message and keeps it for the end-of-run summary."""

    texts: list[str] = field(default_factory=list)
    cards: list[tuple[Any, str]] = field(default_factory=list)
    recoveries: list[tuple[str, int, str]] = field(default_factory=list)

    async def send(self, text: str) -> None:
        self.texts.append(text)
        print(_boxed("Telegram message", text))

    async def send_card(self, card: Any, text: str) -> None:
        self.cards.append((card, text))
        print(_boxed(f"Telegram approval card (with Approve / Reject buttons), expires "
                     f"{card.expires_at:%H:%M} IST", text))

    async def send_recovery(self, book: str, stop_id: int, text: str) -> None:
        self.recoveries.append((book, stop_id, text))
        print(_boxed(f"Telegram recovery plan for {book}", text))


def _boxed(title: str, text: str) -> str:
    body = "\n".join(f"  | {line}" for line in text.splitlines())
    return f"  [{title}]\n{body}"


def _heading(text: str) -> None:
    print(f"\n=== {text} ===")


# ---------------------------------------------------------------- the report sections

def print_news_digest(store: Any) -> None:
    _heading("News digest")
    digest = load_model(store, pipeline.NEWS_DIGEST_KEY, NewsDigest)
    if digest is None:
        print("(none: no relevant news, the AI returned nothing, or the data feed was down)")
        return
    print(f"Market mood: {digest.market_mood}. {digest.market_summary}")
    for s in digest.symbols:
        print(f"  {s.symbol}: sentiment {s.sentiment:+.2f}, {s.materiality} materiality - {s.summary}")
    if not digest.symbols:
        print("  (no stock-specific news)")


def print_shortlist(store: Any) -> None:
    _heading("Shortlist")
    shortlist = store.get_state(pipeline.SHORTLIST_KEY) or []
    print(f"{len(shortlist)} symbol(s): " + (", ".join(shortlist) if shortlist else "(none)"))


def print_ideas(store: Any) -> None:
    _heading("Portfolio manager ideas")
    ideas = [i for i in (store.get_idea(x) for x in store.get_state(pipeline.TODAYS_IDEAS_KEY) or [])
             if i is not None]
    if not ideas:
        print("(no ideas today: the right answer when nothing is clearly worth the costs)")
    for i in ideas:
        print(f"{i.side} {i.symbol}: limit {inr(i.limit_price)}, stop {inr(i.stop)}, target {inr(i.target)}, "
              f"expected move {i.expected_move_pct:.1f}%, confidence {i.confidence:.2f}, "
              f"hold ~{i.holding_days} day(s)\n    reason: {i.reason}")


def print_ai_cost(settings: Settings, store: Any, db_path: Path, now: datetime) -> None:
    _heading("AI cost of this run")
    rate = settings.ai.usd_inr
    total = store.ai_spend_usd_on(now.date())
    try:
        with closing(sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT model, COUNT(*), SUM(input_tokens), SUM(output_tokens), "
                              "SUM(cache_read), SUM(cache_write), SUM(cost_usd) FROM ai_usage "
                              "GROUP BY model ORDER BY model").fetchall()
    except sqlite3.Error as exc:
        print(f"(per-model breakdown unavailable: {type(exc).__name__})")
        rows = []
    for model, calls, inp, out, cache_read, cache_write, cost in rows:
        print(f"  {model}: {calls} call(s), {inp:,} input / {out:,} output tokens, "
              f"{cache_read:,} cache-read / {cache_write:,} cache-write, {usd(cost)}")
    cap = settings.ai.monthly_budget_usd
    print(f"Total: {usd(total)} = {inr(total * rate)} (at {rate:g} INR per US$); "
          f"{total / cap * 100:.1f}% of the {usd(cap)} monthly cap")


# ---------------------------------------------------------------- the run

async def simulate(settings: Settings, notifier: PrintNotifier, *, market_data: Any = None,
                   llm_client: Any = None, http_get: Any = None, news_parse: Any = None,
                   now: datetime | None = None) -> None:
    """premarket, then open_session, once, with ``dry_run=True``. ``market_data``, ``llm_client``,
    ``http_get`` (the Nifty 100 list) and ``news_parse`` (the RSS feeds) default to the real feed, the
    real Claude client and the real network (tests pass fakes)."""
    now = now or now_ist()
    extra = {"http_get": http_get} if http_get is not None else {}
    svc = build_services(settings, notifier, dry_run=True, market_data=market_data,
                         llm_client=llm_client, clock=lambda: now, **extra)
    if news_parse is not None:
        svc.news_parse = news_parse
    try:
        _heading(f"Live dry run, {now:%a %d %b %Y %H:%M} IST (dry-run: no orders, nothing goes to Telegram)")
        print(f"Universe: {len(svc.universe)} symbols. Database: {settings.db_path}")
        if not is_trading_day(now.date(), settings.holidays):
            print("Note: today is not an NSE trading day, so quotes are stale and no cards will be made.")

        _heading("Step 1: premarket (news, indicators, shortlist, AI ideas)")
        await pipeline.premarket(svc, now)
        print_news_digest(svc.store)
        print_shortlist(svc.store)
        print_ideas(svc.store)

        _heading("Step 2: open_session (sizing and approval cards)")
        await pipeline.open_session(svc, now)
        cards = len(notifier.cards)
        print(f"\n{cards} approval card(s) would have been sent.")
        if not cards:
            risk = settings.risk
            if not (is_trading_day(now.date(), settings.holidays)
                    and in_window(now, risk.entry_start, risk.entry_end)):
                print(f"No cards: the market is closed now (entries run {risk.entry_start}-{risk.entry_end} IST "
                      "on trading days) and the free quotes are stale. Run again on a trading day between "
                      "09:30 and 15:10.")
            else:
                print("No cards: no idea had a fresh quote and a book that could take it "
                      "(see the log lines above for each skipped idea).")
        print_ai_cost(settings, svc.store, Path(settings.db_path), now)
    finally:
        svc.store.close()


def _fresh_database(path: Path, real_db: Path) -> None:
    """Delete the previous simulate database (and its WAL files), never the real one."""
    if path.resolve() == real_db.resolve():
        raise SystemExit(f"Refusing to use {path}: it is the real database.")
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):           # the rupee sign on a Windows console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Live dry run of premarket + open_session (real data, real "
                                                 "Claude API, no orders, nothing sent to Telegram).")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    config_path = Path(parser.parse_args(argv).config).resolve()

    load_dotenv(config_path.parent / ".env")           # before the Anthropic client is built
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(NO_KEY_MESSAGE)
        return 2
    if not config_path.is_file():
        print(f"Config file not found: {config_path}")
        return 2
    try:
        settings = load_settings(config_path, env=_DUMMY_TELEGRAM_ENV)
    except ValidationError as exc:
        print(f"{config_path} has {len(exc.errors())} problem(s): "
              + "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors(include_input=False)))
        return 2
    real_db = settings.db_path if settings.db_path.is_absolute() else config_path.parent / settings.db_path
    sim_db = real_db.parent / SIMULATE_DB_NAME
    _fresh_database(sim_db, real_db)
    settings = settings.model_copy(update={"db_path": sim_db})

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("aitrader").setLevel(logging.INFO)
    try:
        asyncio.run(simulate(settings, PrintNotifier()))
    except Exception as exc:  # noqa: BLE001 - no traceback: it could show request details; type and place only
        print(f"\nThe dry run failed: {describe_error(exc)}. See the log lines above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
