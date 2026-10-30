"""Wiring: ``Services`` (everything the jobs and the bot share), the job guard and the schedule.

``build_services`` is restart-safe: cash, the loss-limit baselines and ``phase1_start`` are seeded
only on the first run (``phase1_start`` never in a dry run). ``schedule_jobs`` registers the jobs of
``pipeline.py`` on PTB's JobQueue at the IST times of ``ScheduleConfig``; every job runs through
``guarded``.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import httpx

from aitrader import pipeline
from aitrader.agents.llm import LLMClient
from aitrader.approvals import ApprovalBook
from aitrader.broker import PaperBroker
from aitrader.calendar import IST, is_trading_day, now_ist
from aitrader.config import Settings
from aitrader.control import JOBS
from aitrader.data import BROWSER_USER_AGENT
from aitrader.data.bhavcopy import fetch_closes
from aitrader.data.market_data import MarketData, YFinanceMarketData
from aitrader.data.news import http_parse
from aitrader.data.screener import load_universe
from aitrader.execution import Executor, describe_error
from aitrader.exits import ExitWatcher
from aitrader.pipeline_support import notify_once
from aitrader.risk.limits import LossLimits
from aitrader.risk.manager import RiskManager
from aitrader.store import Store

log = logging.getLogger(__name__)

PHASE1_START_KEY = "phase1_start"
_PTB_DAYS = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")  # PTB: 0 = Sunday

JobFn = Callable[[Any, datetime], Awaitable[None]]
JOB_FUNCS: dict[str, JobFn] = {
    "premarket": pipeline.premarket,
    "open_session": pipeline.open_session,
    "intraday": pipeline.intraday,
    "exits": pipeline.exits_job,
    "postmarket": pipeline.postmarket,
    "weekly": pipeline.weekly,
}
assert tuple(JOB_FUNCS) == JOBS, "job names must match control.JOBS (shown by /status)"


@dataclass
class Services:
    settings: Settings
    store: Store
    market_data: MarketData
    broker: PaperBroker
    limits: dict[str, LossLimits]
    risk: RiskManager
    executor: Executor
    exits: ExitWatcher
    approvals: ApprovalBook
    llm: LLMClient
    notifier: Any
    universe: list[str]
    aliases: dict[str, list[str]]
    dry_run: bool
    # Network seams, replaceable in tests.
    news_parse: Callable[[str], Any] = field(default=http_parse)
    fetch_closes: Callable[[date], dict[str, float]] = field(default=fetch_closes)


class LateBoundNotifier:
    """A notifier whose target (the Telegram bot) is attached after the services exist: the PTB
    application needs the services to be built, and the services need a notifier."""

    def __init__(self) -> None:
        self.target: Any = None

    def _require(self) -> Any:
        if self.target is None:
            raise RuntimeError("notifier used before the Telegram bot was attached")
        return self.target

    async def send(self, text: str) -> None:
        await self._require().send(text)

    async def send_card(self, card: Any, text: str) -> None:
        await self._require().send_card(card, text)

    async def send_recovery(self, book: str, stop_id: int, text: str) -> None:
        target = self._require()
        sender = getattr(target, "send_recovery", None)
        await (sender(book, stop_id, text) if callable(sender) else target.send(text))


# ---------------------------------------------------------------- universe

def universe_cache_path(settings: Settings) -> Path:
    return Path(settings.db_path).parent / "cache" / "nifty100.csv"


def load_universe_cached(settings: Settings, http_get: Callable = httpx.get) -> tuple[list[str], dict[str, list[str]]]:
    """Nifty 100 + ETFs. Downloads the CSV and caches it; on failure uses the cache; with no cache,
    the ETFs only (with a warning)."""
    etfs = settings.data.etfs
    cache = universe_cache_path(settings)
    try:
        resp = http_get(settings.data.nifty100_csv_url, headers={"User-Agent": BROWSER_USER_AGENT},
                        timeout=30, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        symbols, aliases = load_universe(text, etfs)
        if len(symbols) <= len(set(etfs)):
            raise ValueError("the Nifty 100 CSV had no stocks")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
        return symbols, aliases
    except Exception as exc:  # noqa: BLE001 - NSE often blocks scripts; the cache covers it
        log.warning("Nifty 100 download failed (%s); trying the cached copy", describe_error(exc))
    if cache.exists():
        return load_universe(cache.read_text(encoding="utf-8"), etfs)
    log.warning("no cached Nifty 100 list at %s: trading the ETFs only (%s)", cache, ", ".join(etfs))
    return load_universe("", etfs)


# ---------------------------------------------------------------- services

def build_services(settings: Settings, notifier: Any, dry_run: bool, market_data: MarketData | None = None,
                   llm_client: LLMClient | None = None, *, http_get: Callable = httpx.get,
                   clock: Callable[[], datetime] | None = None) -> Services:
    clock = clock or now_ist
    store = Store(settings.db_path)
    with store.transaction():                 # first run only: a restart never re-seeds
        for book in settings.books:
            if not store.has_cash(book.name):
                store.set_cash(book.name, float(book.capital))
                log.info("seeded %s with cash %.2f", book.name, book.capital)
        # A dry run never starts the Phase 1 clock (it has its own database; this is defence in depth).
        if store.get_state(PHASE1_START_KEY) is None and not dry_run:
            store.set_state(PHASE1_START_KEY, clock().astimezone(IST).date().isoformat())
    limits = {b.name: LossLimits(store, b, settings.holidays) for b in settings.books}   # seeds baselines once

    market_data = market_data if market_data is not None else YFinanceMarketData()
    universe, aliases = load_universe_cached(settings, http_get)
    risk = RiskManager(store, settings, limits, set(universe))
    broker = PaperBroker(store, settings.charges)
    executor = Executor(store, broker, risk, notifier, dry_run,
                        min_interval_s=1.0 / settings.risk.max_orders_per_sec)
    names = [b.name for b in settings.books]
    return Services(
        settings=settings, store=store, market_data=market_data, broker=broker, limits=limits, risk=risk,
        executor=executor, exits=ExitWatcher(store, executor, market_data, settings.risk, names),
        approvals=ApprovalBook(store, settings.approval, settings.risk.entry_end),
        llm=llm_client if llm_client is not None else LLMClient(store, settings.ai),
        notifier=notifier, universe=universe, aliases=aliases, dry_run=dry_run,
    )


# ---------------------------------------------------------------- jobs

def guarded(svc: Any, name: str, fn: JobFn, clock: Callable[[], datetime] = now_ist,
            trading_days_only: bool = True) -> Callable[[Any], Awaitable[None]]:
    """``fn`` as a PTB job callback: skipped on non-trading days (unless ``trading_days_only`` is
    False), errors logged by type and place only and sent to the owner, ``last_run:<name>`` on success."""
    async def job(context: Any = None) -> None:
        now = clock().astimezone(IST)
        if trading_days_only and not is_trading_day(now.date(), svc.settings.holidays):
            return
        try:
            await fn(svc, now)
        except Exception as exc:  # noqa: BLE001 - a job failure must reach the owner, not kill the bot
            where = describe_error(exc)
            log.error("job %s failed: %s", name, where)       # logged every time, alerted once a day
            try:
                await notify_once(svc, f"job_error:{name}:{type(exc).__name__}:{now.date().isoformat()}",
                                  f"Job {name} failed: {where}. See the log; the next run will try again.")
            except Exception as send_exc:  # noqa: BLE001
                log.error("could not tell the owner: %s", describe_error(send_exc))
            return
        svc.store.set_state(f"last_run:{name}", now.isoformat())

    job.__name__ = f"job_{name}"
    return job


def _ist_time(hhmm: str) -> time:
    hours, minutes = (int(x) for x in hhmm.split(":"))
    return time(hours, minutes, tzinfo=IST)


def schedule_jobs(app: Any, svc: Any, clock: Callable[[], datetime] = now_ist) -> None:
    jq = app.job_queue
    if jq is None:
        raise RuntimeError("PTB has no JobQueue: install python-telegram-bot[job-queue]")
    s = svc.settings.schedule

    def daily(name: str, hhmm: str, **kw: Any) -> None:
        jq.run_daily(guarded(svc, name, JOB_FUNCS[name], clock), time=_ist_time(hhmm), name=name, **kw)

    daily("premarket", s.premarket)
    daily("open_session", s.open_session)
    for hhmm in s.intraday:
        daily("intraday", hhmm)
    daily("postmarket", s.postmarket)
    jq.run_repeating(guarded(svc, "exits", JOB_FUNCS["exits"], clock),
                     interval=svc.settings.risk.exit_check_minutes * 60, name="exits")
    jq.run_daily(guarded(svc, "weekly", JOB_FUNCS["weekly"], clock, trading_days_only=False),
                 time=_ist_time(s.weekly_time), days=(_PTB_DAYS.index(s.weekly_day),), name="weekly")
