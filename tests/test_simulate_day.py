"""scripts/simulate_day.py, offline: the no-key exit, the safeguards (own database, no Telegram values,
no secrets in output) and a full run of ``simulate`` against fakes. The live run itself needs the owner's
Anthropic key and is not tested here."""

import importlib.util
import os
import shutil
import time
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pytest
from pipeline_fakes import REPO_CONFIG, SETTINGS, FakeLLM, FakeMarketData, csv_get

from aitrader.agents.schemas import (
    IdeaOut,
    NewsDigest,
    PMDecision,
    SymbolNews,
    TechnicalBatch,
    TechnicalView,
)
from aitrader.calendar import at
from aitrader.models import OrderStatus
from aitrader.store import Store

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "simulate_day.py"
MON = date(2026, 11, 2)
NOW = at(MON, "09:50")                      # inside the entry window, the feed has caught up


def load_script():
    spec = importlib.util.spec_from_file_location("simulate_day_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated_environ():
    """``main()`` calls ``load_dotenv``, which writes the process environment. Whatever a test (or the
    script) adds or changes there is undone afterwards, so no key or Telegram value leaks into other tests."""
    with mock.patch.dict(os.environ):
        yield


@pytest.fixture
def sim():
    return load_script()


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """A copy of config.yaml in an empty folder (so no real .env is found) and no key in the environment."""
    shutil.copy(REPO_CONFIG, tmp_path / "config.yaml")
    for name in ("ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_OWNER_CHAT_ID"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


# ---------------------------------------------------------------- no key

def test_without_a_key_it_says_so_and_exits_2_before_building_anything(sim, config_dir, capsys):
    assert sim.main(["--config", str(config_dir / "config.yaml")]) == 2
    out = capsys.readouterr().out
    assert "Set ANTHROPIC_API_KEY in .env to run the live dry run" in out
    assert not (config_dir / "data").exists()               # nothing was built, no database made


def test_an_empty_key_in_dotenv_counts_as_missing(sim, config_dir, capsys):
    (config_dir / ".env").write_text("ANTHROPIC_API_KEY=\nTELEGRAM_BOT_TOKEN=\n", encoding="utf-8")
    assert sim.main(["--config", str(config_dir / "config.yaml")]) == 2
    assert "Set ANTHROPIC_API_KEY in .env" in capsys.readouterr().out


def test_the_key_is_read_from_dotenv_next_to_the_config_and_never_printed(sim, config_dir, capsys, monkeypatch):
    (config_dir / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-SECRET-VALUE\n", encoding="utf-8")

    async def fake_simulate(settings, notifier, **kw):
        pass

    monkeypatch.setattr(sim, "simulate", fake_simulate)
    assert sim.main(["--config", str(config_dir / "config.yaml")]) == 0
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-SECRET-VALUE"      # it came from that .env
    assert "SECRET-VALUE" not in capsys.readouterr().out


# ---------------------------------------------------------------- safeguards

def test_main_uses_its_own_database_and_dummy_telegram_values(sim, config_dir, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "real-bot-token-123")       # must not even be read
    monkeypatch.setenv("TELEGRAM_OWNER_CHAT_ID", "987654")
    real_db = config_dir / "data" / "aitrader.db"
    real_db.parent.mkdir()
    real_db.write_bytes(b"the real database")
    (config_dir / "data" / "simulate.db").write_bytes(b"a previous run")
    seen = {}

    async def fake_simulate(settings, notifier, **kw):
        seen["settings"], seen["notifier"] = settings, notifier

    monkeypatch.setattr(sim, "simulate", fake_simulate)
    assert sim.main(["--config", str(config_dir / "config.yaml")]) == 0

    settings = seen["settings"]
    assert settings.db_path == config_dir / "data" / "simulate.db"
    assert not settings.db_path.exists()                    # the previous run's database was removed
    assert real_db.read_bytes() == b"the real database"     # and the real one was never touched
    assert settings.telegram_bot_token.get_secret_value() != "real-bot-token-123"
    assert settings.telegram_owner_chat_id != 987654
    assert isinstance(seen["notifier"], sim.PrintNotifier)


def test_fresh_database_removes_the_wal_files_and_refuses_the_real_database(sim, tmp_path):
    simulate_db, real_db = tmp_path / "simulate.db", tmp_path / "aitrader.db"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{simulate_db}{suffix}").write_bytes(b"x")
    real_db.write_bytes(b"real")
    sim._fresh_database(simulate_db, real_db)
    assert not list(tmp_path.glob("simulate.db*")) and real_db.read_bytes() == b"real"
    with pytest.raises(SystemExit):
        sim._fresh_database(real_db, real_db)
    assert real_db.read_bytes() == b"real"


def test_a_failed_run_prints_the_error_type_and_place_never_its_message(sim, config_dir, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")

    async def failing_simulate(settings, notifier, **kw):
        raise RuntimeError("secret-token-123 https://example.invalid/?key=abc")

    monkeypatch.setattr(sim, "simulate", failing_simulate)
    assert sim.main(["--config", str(config_dir / "config.yaml")]) == 1
    out = capsys.readouterr().out
    assert "RuntimeError" in out and "secret-token-123" not in out and "example.invalid" not in out


# ---------------------------------------------------------------- a full run against fakes

class UsageLLM(FakeLLM):
    """FakeLLM that also records a billed call (US$0.50) in the database, as the real client does."""

    def __init__(self, db_path, ts, responses):
        super().__init__(responses)
        self.db_path, self.ts = db_path, ts

    def run(self, agent, system, user, output_model, max_tokens=16000):
        store = Store(self.db_path)
        try:
            store.add_ai_usage("claude-opus-5-5", 1000, 200, 0, 0, 0.5, self.ts)
        finally:
            store.close()
        return super().run(agent, system, user, output_model, max_tokens)


def llm_responses():
    return {
        "NewsDigest": NewsDigest(market_mood="bullish", market_summary="Banks lead the market", symbols=[
            SymbolNews(symbol="NIFTYBEES", sentiment=0.4, materiality="medium", summary="index firm")]),
        "TechnicalBatch": TechnicalBatch(views=[TechnicalView(
            symbol="NIFTYBEES", trend="up", support=270, resistance=300, signal="buy", confidence=0.6,
            notes="ok")]),
        "PMDecision": PMDecision(market_view="calm", ideas=[IdeaOut(
            symbol="NIFTYBEES", side="BUY", limit_price=280.0, stop=266.0, target=302.4, expected_move_pct=8.0,
            confidence=0.7, holding_days=5, reason="Index breakout above 279 on rising volume")]),
    }


def news_parse(url):
    published = time.struct_time((2026, 11, 2, 2, 0, 0, 0, 0, 0))                 # 07:30 IST
    return {"feed": {"title": "F"}, "bozo": 0, "entries": [
        {"title": "Nifty opens higher as banks rally", "link": "https://n/1", "summary": "",
         "published_parsed": published}]}


async def test_simulate_prints_the_whole_morning_and_places_nothing(sim, tmp_path, capsys):
    db = tmp_path / "data" / "simulate.db"
    settings = SETTINGS.model_copy(update={
        "db_path": db, "data": SETTINGS.data.model_copy(update={"rss_feeds": ["https://feed"]})})
    md = FakeMarketData({"NIFTYBEES": 280.0, "ITC": 400.0}, NOW - timedelta(minutes=16))
    notifier = sim.PrintNotifier()

    await sim.simulate(settings, notifier, market_data=md, llm_client=UsageLLM(db, NOW, llm_responses()),
                       http_get=csv_get(), news_parse=news_parse, now=NOW)

    out = capsys.readouterr().out
    assert "Step 1: premarket" in out and "Step 2: open_session" in out
    assert "Market mood: bullish. Banks lead the market" in out and "NIFTYBEES: sentiment +0.40" in out
    assert "Shortlist" in out and "NIFTYBEES" in out
    assert "BUY NIFTYBEES" in out and "reason: Index breakout above 279 on rising volume" in out
    assert "1 approval card(s) would have been sent." in out
    assert "Telegram approval card" in out and "Approve" in out
    assert "AI cost of this run" in out and "claude-opus-5-5: 3 call(s), 3,000 input / 600 output tokens" in out
    assert "Total: US$1.50 = ₹132.00" in out
    assert len(notifier.cards) == 1 and set(notifier.cards[0][0].sizes) == {"realistic", "learning"}

    store = Store(db)                                       # dry-run: not a single order, fill or position
    try:
        assert all(store.orders_with_status(status) == [] for status in OrderStatus)
        assert store.cash("realistic") == 1000.0 and store.cash("learning") == 100000.0
        assert store.positions("realistic") == [] and store.positions("learning") == []
    finally:
        store.close()


async def test_simulate_on_a_closed_day_explains_why_there_are_no_cards(sim, tmp_path, capsys):
    db = tmp_path / "data" / "simulate.db"
    settings = SETTINGS.model_copy(update={"db_path": db})
    saturday = at(date(2026, 11, 7), "11:00")
    md = FakeMarketData({"NIFTYBEES": 280.0}, at(date(2026, 11, 6), "15:29"))      # Friday's last tick

    await sim.simulate(settings, sim.PrintNotifier(), market_data=md,
                       llm_client=UsageLLM(db, saturday, llm_responses()), http_get=csv_get(),
                       news_parse=news_parse, now=saturday)

    out = capsys.readouterr().out
    assert "not an NSE trading day" in out
    assert "0 approval card(s) would have been sent." in out
    assert "No cards: the market is closed now" in out
