"""Services wiring, the job guard, the schedule and the CLI (offline)."""

import os
import subprocess
import sys
from datetime import date, timedelta

from pipeline_fakes import (LEARN, REAL, REPO_CONFIG, SETTINGS, FakeLLM, FakeMarketData, RecordingNotifier,
                            csv_get, failing_get, make_svc)
from telegram.ext import Application

from aitrader import app as app_mod
from aitrader.app import Services, guarded, schedule_jobs
from aitrader.calendar import at
from aitrader.control import JOBS

MON = date(2026, 11, 2)
HOLIDAY = date(2026, 11, 10)          # Diwali-Balipratipada, a Tuesday
SAT = date(2026, 11, 7)


def md(now):
    return FakeMarketData({"NIFTYBEES": 280.0}, now)


# ---------------------------------------------------------------- build_services

def test_build_services_seeds_capital_once(tmp_path):
    now = at(MON, "08:00")
    svc = make_svc(tmp_path, md(now), now=now)
    assert isinstance(svc, Services)
    assert svc.store.cash(REAL) == 1000.0 and svc.store.cash(LEARN) == 100000.0
    assert svc.limits[REAL].baseline() == 1000.0
    assert svc.store.get_state("phase1_start") == MON.isoformat()
    assert svc.universe[:2] == ["RELIANCE", "ITC"] and "NIFTYBEES" in svc.universe
    assert svc.aliases["RELIANCE"][0] == "Reliance Industries Ltd."
    assert svc.notifier is not None and svc.dry_run is False

    svc.store.set_cash(REAL, 900.0)
    svc.store.set_state("baseline:realistic", 950.0)
    svc.store.close()

    later = at(MON + timedelta(days=1), "08:00")
    again = make_svc(tmp_path, md(later), now=later)
    assert again.store.cash(REAL) == 900.0                         # never re-seeded on a restart
    assert again.limits[REAL].baseline() == 950.0
    assert again.store.get_state("phase1_start") == MON.isoformat()
    again.store.close()


def test_universe_cache_and_etf_fallback(tmp_path, caplog):
    now = at(MON, "08:00")
    cache = tmp_path / "data" / "cache" / "nifty100.csv"

    svc = make_svc(tmp_path / "a", md(now), now=now, http_get=failing_get)
    assert svc.universe == SETTINGS.data.etfs                     # no download, no cache: ETFs only
    assert "ETFs only" in caplog.text and "secret" not in caplog.text
    svc.store.close()

    svc = make_svc(tmp_path, md(now), now=now, http_get=csv_get())
    assert cache.exists()                                           # a good download is cached
    svc.store.close()

    svc = make_svc(tmp_path, md(now), now=now, http_get=failing_get)
    assert "RELIANCE" in svc.universe                               # download failed: cached copy used
    svc.store.close()


def test_build_services_dry_run_reaches_executor(tmp_path):
    now = at(MON, "08:00")
    settings = SETTINGS.model_copy(update={"db_path": tmp_path / "x.db"})
    svc = app_mod.build_services(settings, RecordingNotifier(), dry_run=True, market_data=md(now),
                                 llm_client=FakeLLM(), http_get=csv_get(), clock=lambda: now)
    assert svc.dry_run is True and svc.executor._dry_run is True
    svc.store.close()


# ---------------------------------------------------------------- guard

async def test_jobs_noop_on_holiday(tmp_path):
    calls = []

    async def fn(svc, now):
        calls.append(now)

    holiday_now = at(HOLIDAY, "10:00")
    svc = make_svc(tmp_path, md(holiday_now), now=holiday_now)
    await guarded(svc, "intraday", fn, clock=lambda: holiday_now)(None)
    await guarded(svc, "exits", fn, clock=lambda: at(SAT, "10:00"))(None)
    assert calls == []
    assert svc.store.get_state("last_run:intraday") is None

    await guarded(svc, "weekly", fn, clock=lambda: at(SAT, "10:00"), trading_days_only=False)(None)
    trading = at(MON, "10:20")
    await guarded(svc, "intraday", fn, clock=lambda: trading)(None)
    assert calls == [at(SAT, "10:00"), trading]
    assert svc.store.get_state("last_run:intraday") == trading.isoformat()
    assert svc.store.get_state("last_run:weekly") == at(SAT, "10:00").isoformat()


async def test_guard_catches_and_notifies(tmp_path, caplog):
    async def boom(svc, now):
        raise RuntimeError("token=secret-123")

    now = at(MON, "10:20")
    svc = make_svc(tmp_path, md(now), now=now)
    await guarded(svc, "postmarket", boom, clock=lambda: now)(None)
    assert svc.store.get_state("last_run:postmarket") is None
    assert len(svc.notifier.texts) == 1 and "postmarket" in svc.notifier.texts[0]
    assert "RuntimeError" in svc.notifier.texts[0]
    assert "secret" not in svc.notifier.texts[0] and "secret" not in caplog.text


# ---------------------------------------------------------------- schedule

def test_schedule_jobs_registers_every_job(tmp_path):
    now = at(MON, "08:00")
    svc = make_svc(tmp_path, md(now), now=now)
    application = Application.builder().token("123:abc").build()
    schedule_jobs(application, svc)
    jobs = application.job_queue.jobs()
    names = [j.name for j in jobs]
    assert set(names) == set(JOBS)
    assert names.count("intraday") == len(SETTINGS.schedule.intraday)
    triggers = {j.name: j.trigger for j in jobs}
    assert "day_of_week='sat'" in str(triggers["weekly"]).replace('"', "'")
    assert str(triggers["premarket"].timezone) == "Asia/Kolkata" and "hour='8'" in str(triggers["premarket"])
    assert "0:03:00" in str(triggers["exits"])


# ---------------------------------------------------------------- CLI

def _env(**extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_OWNER_CHAT_ID", "ANTHROPIC_API_KEY")}
    env.update(extra)
    return env


def test_check_config_cli(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "aitrader", "check-config", "--config", str(REPO_CONFIG)],
        env=_env(TELEGRAM_BOT_TOKEN="123:dummy", TELEGRAM_OWNER_CHAT_ID="42"),
        capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "realistic" in out and "learning" in out
    assert "08:30" in out and "Saturday" in out
    assert f"{len(SETTINGS.holidays)} holidays" in out
    assert "123:dummy" not in out


def test_check_config_cli_missing_telegram_env(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "aitrader", "check-config", "--config", str(config)],
                            env=_env(), capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode != 0
    assert "TELEGRAM_BOT_TOKEN" in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
