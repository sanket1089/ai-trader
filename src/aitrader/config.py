"""Typed settings: non-secret values from ``config.yaml``, secrets from ``.env``/environment."""

import os
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr, StringConstraints

# Quoted "HH:MM" IST wall-clock time, 24-hour.
HHMM = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
Weekday = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class _Config(BaseModel):
    # A misspelled key in config.yaml must fail loudly rather than be silently ignored.
    model_config = ConfigDict(extra="forbid")


class ChargesConfig(_Config):
    """Delivery charge rates. ``*_pct`` are percent of turnover; the rest are flat INR."""

    brokerage_per_order: float
    stt_pct: float
    stamp_buy_pct: float
    exchange_pct: float
    sebi_pct: float
    gst_pct: float
    dp_per_sell: float


class BookConfig(_Config):
    name: str
    capital: float
    daily_loss_limit: float
    total_loss_limit: float
    max_entries_per_day: int
    max_sell_days_per_week: int | None = None  # None: no weekly cap
    max_position_pct: float  # percent of book equity


class RiskConfig(_Config):
    entry_band_pct: float
    exit_band_pct: float
    cost_multiple: float
    data_delay_minutes: int
    stale_extra_minutes: int
    entry_start: HHMM
    entry_end: HHMM
    exit_start: HHMM
    max_orders_per_sec: float
    exit_check_minutes: int
    exit_reprice_minutes: int
    exit_max_reprices: int


class ApprovalConfig(_Config):
    expiry_minutes: int
    reprice_tolerance_pct: float


class AgentModel(_Config):
    model: str
    effort: str


class ModelPrice(_Config):
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float


class AIConfig(_Config):
    monthly_budget_usd: float
    usd_inr: float
    reduced_at_pct: float = 80
    agents: dict[str, AgentModel]
    prices: dict[str, ModelPrice]


class DataConfig(_Config):
    etfs: list[str]
    nifty100_csv_url: str
    rss_feeds: list[str]
    shortlist_size: int


class ScheduleConfig(_Config):
    premarket: HHMM
    open_session: HHMM
    intraday: list[HHMM]
    postmarket: HHMM
    weekly_day: Weekday
    weekly_time: HHMM


class Settings(_Config):
    # Validation errors would otherwise echo the whole input, including the bot token.
    model_config = ConfigDict(hide_input_in_errors=True)

    books: list[BookConfig]
    charges: ChargesConfig
    risk: RiskConfig
    approval: ApprovalConfig
    ai: AIConfig
    data: DataConfig
    schedule: ScheduleConfig
    holidays: list[date]
    db_path: Path
    telegram_owner_chat_id: int
    telegram_bot_token: SecretStr


_ENV_FIELDS = {
    "TELEGRAM_OWNER_CHAT_ID": "telegram_owner_chat_id",
    "TELEGRAM_BOT_TOKEN": "telegram_bot_token",
}


def load_settings(config_path: Path, env: Mapping[str, str] | None = None) -> Settings:
    """Load ``config_path`` and overlay the secrets from ``env``.

    With ``env=None`` the secrets come from the ``.env`` file next to ``config_path``,
    overridden by real environment variables. An unset or empty secret is treated as
    missing and raises ``pydantic.ValidationError``.
    """
    config_path = Path(config_path)
    if env is None:
        env = {**dotenv_values(config_path.parent / ".env"), **os.environ}

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for env_name, field in _ENV_FIELDS.items():
        if env.get(env_name):
            data[field] = env[env_name]
    return Settings.model_validate(data)
