import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aitrader.config import load_settings

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
SECRET_TOKEN = "s3cr3t-bot-token-value"
ENV = {"TELEGRAM_OWNER_CHAT_ID": "42", "TELEGRAM_BOT_TOKEN": SECRET_TOKEN}


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    dest = tmp_path / "config.yaml"
    shutil.copy(REPO_CONFIG, dest)
    return dest


def test_load_settings_reads_yaml_and_env(config_path):
    s = load_settings(config_path, env=ENV)
    books = {b.name: b for b in s.books}
    assert books["realistic"].capital == 1000
    assert books["realistic"].daily_loss_limit == 100
    assert books["realistic"].total_loss_limit == 500
    assert books["realistic"].max_entries_per_day == 1
    assert books["realistic"].max_sell_days_per_week == 3
    assert books["realistic"].max_position_pct == 100
    assert books["learning"].capital == 100000
    assert books["learning"].max_entries_per_day == 3
    assert books["learning"].max_sell_days_per_week is None
    assert books["learning"].max_position_pct == 20
    assert s.ai.monthly_budget_usd == 100
    assert s.ai.usd_inr == 88
    assert s.ai.reduced_at_pct == 80
    assert s.ai.agents["portfolio_manager"].model == "claude-opus-5-5"
    assert s.ai.agents["portfolio_manager"].effort == "medium"
    assert s.ai.agents["news"].effort == "low"
    assert s.ai.prices["claude-opus-5-5"].cache_read == 0.20
    assert s.telegram_owner_chat_id == 42
    assert s.telegram_bot_token.get_secret_value() == SECRET_TOKEN


@pytest.mark.parametrize("model", ["claude-opus-5", "claude-opus-4-8"])
def test_fallback_models_have_price_rows(config_path, model):
    price = load_settings(config_path, env=ENV).ai.prices[model]
    assert (price.input, price.output, price.cache_read, price.cache_write) == (5, 25, 0.50, 6.25)


def test_repo_config_values_from_global_constraints(config_path):
    s = load_settings(config_path, env=ENV)
    assert s.risk.entry_band_pct == 0.5
    assert s.risk.exit_band_pct == 1.0
    assert s.risk.cost_multiple == 3
    assert (s.risk.entry_start, s.risk.entry_end, s.risk.exit_start) == ("09:20", "15:10", "09:15")
    assert s.risk.max_orders_per_sec == 1
    assert (s.risk.exit_check_minutes, s.risk.exit_reprice_minutes, s.risk.exit_max_reprices) == (3, 2, 3)
    assert s.approval.expiry_minutes == 10
    assert s.approval.reprice_tolerance_pct == 0.5
    assert s.charges.dp_per_sell == 15.34
    assert s.data.etfs == ["NIFTYBEES", "GOLDBEES", "LIQUIDBEES", "BANKBEES"]
    assert s.data.shortlist_size == 25
    assert s.schedule.intraday == ["10:20", "11:20", "12:20", "13:20", "14:20"]
    assert (s.schedule.weekly_day, s.schedule.weekly_time) == ("Saturday", "10:00")
    assert date(2026, 1, 26) in s.holidays
    assert all(isinstance(d, date) for d in s.holidays)


def test_missing_owner_chat_id_raises(config_path):
    with pytest.raises(ValidationError):
        load_settings(config_path, env={"TELEGRAM_BOT_TOKEN": "x"})


def test_validation_error_does_not_leak_token(config_path):
    with pytest.raises(ValidationError) as exc:
        load_settings(config_path, env={"TELEGRAM_BOT_TOKEN": SECRET_TOKEN})
    assert SECRET_TOKEN not in str(exc.value)


def test_empty_secret_values_count_as_missing(config_path):
    with pytest.raises(ValidationError):
        load_settings(config_path, env={"TELEGRAM_OWNER_CHAT_ID": "", "TELEGRAM_BOT_TOKEN": ""})


def test_secret_not_in_repr(config_path):
    s = load_settings(config_path, env=ENV)
    assert SECRET_TOKEN not in repr(s)
    assert SECRET_TOKEN not in str(s)
    assert SECRET_TOKEN not in s.model_dump_json()


def test_env_none_merges_dotenv_file_with_os_environ_winning(config_path, monkeypatch):
    (config_path.parent / ".env").write_text("TELEGRAM_OWNER_CHAT_ID=7\nTELEGRAM_BOT_TOKEN=from-file\n")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "from-os")
    monkeypatch.delenv("TELEGRAM_OWNER_CHAT_ID", raising=False)
    s = load_settings(config_path)
    assert s.telegram_owner_chat_id == 7
    assert s.telegram_bot_token.get_secret_value() == "from-os"


def test_unknown_config_key_is_rejected(config_path):
    data = yaml.safe_load(config_path.read_text())
    data["risk"]["entry_band"] = 0.5   # typo of entry_band_pct
    config_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        load_settings(config_path, env=ENV)


def test_bad_time_format_is_rejected(config_path):
    data = yaml.safe_load(config_path.read_text())
    data["schedule"]["premarket"] = "8:30am"
    config_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        load_settings(config_path, env=ENV)
