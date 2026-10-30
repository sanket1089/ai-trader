"""Command line: ``py -m aitrader run [--dry-run] [--config config.yaml]`` and ``py -m aitrader check-config``.

``run --dry-run`` uses its own database, ``dryrun.db`` next to ``db_path``; the database in use is
printed at startup.

Secrets come from ``.env`` next to the config file (or the real environment) and are never printed.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from aitrader.config import Settings, load_settings

_ENV_NAMES = {"telegram_bot_token": "TELEGRAM_BOT_TOKEN", "telegram_owner_chat_id": "TELEGRAM_OWNER_CHAT_ID"}


def _load(config_path: Path) -> Settings | None:
    """Settings, or ``None`` after printing a plain explanation (no traceback, no input values)."""
    if not config_path.is_file():
        print(f"Config file not found: {config_path}")
        return None
    try:
        settings = load_settings(config_path)
    except ValidationError as exc:
        errors = exc.errors(include_input=False)
        missing = sorted({_ENV_NAMES[e["loc"][0]] for e in errors if e["loc"] and e["loc"][0] in _ENV_NAMES})
        if missing:
            print(f"Missing {', '.join(missing)}: put them in {config_path.parent / '.env'} "
                  "(copy .env.example) or set them in the environment.")
        others = [e for e in errors if not (e["loc"] and e["loc"][0] in _ENV_NAMES)]
        if others:
            print(f"{config_path} has {len(others)} problem(s):")
            for e in others:
                print(f"  {'.'.join(str(p) for p in e['loc'])}: {e['msg']}")
        return None
    except Exception as exc:  # noqa: BLE001 - e.g. YAML syntax errors
        print(f"Could not read {config_path}: {type(exc).__name__}: {exc}")
        return None
    # A relative db_path is relative to the config file, not to wherever the command was run.
    if not settings.db_path.is_absolute():
        settings = settings.model_copy(update={"db_path": config_path.parent / settings.db_path})
    return settings


DRY_RUN_DB_NAME = "dryrun.db"


def settings_for_run(settings: Settings, dry_run: bool) -> Settings:
    """A dry run gets its own database, ``dryrun.db`` next to ``db_path``: it must never seed the
    Phase 1 start or the benchmark, tick the go-live flags, or write snapshots, lessons or orders
    into the real paper-trading database."""
    if not dry_run:
        return settings
    return settings.model_copy(update={"db_path": settings.db_path.parent / DRY_RUN_DB_NAME})


def check_config(config_path: Path) -> int:
    settings = _load(config_path)
    if settings is None:
        return 2
    s = settings.schedule
    holidays = sorted(settings.holidays)
    lines = [f"Config OK: {config_path} (nothing was connected to)", "Books:"]
    for b in settings.books:
        weekly = "no weekly sell-day cap" if b.max_sell_days_per_week is None else \
            f"max {b.max_sell_days_per_week} sell days/week"
        lines.append(f"  {b.name}: capital ₹{b.capital:,.0f}, daily loss limit ₹{b.daily_loss_limit:,.0f}, "
                     f"total loss limit ₹{b.total_loss_limit:,.0f}, max {b.max_entries_per_day} entries/day, "
                     f"{weekly}, max position {b.max_position_pct:g}% of equity")
    lines += [
        "Schedule (IST, trading days):",
        f"  premarket {s.premarket}, open session {s.open_session}, intraday {', '.join(s.intraday)}, "
        f"postmarket {s.postmarket}",
        f"  exits every {settings.risk.exit_check_minutes} min from {settings.risk.exit_start} to 15:30",
        f"  weekly report {s.weekly_day} {s.weekly_time}",
        f"{len(holidays)} holidays" + (f" ({holidays[0]} to {holidays[-1]})" if holidays else ""),
        f"AI budget: US${settings.ai.monthly_budget_usd:g}/month (₹{settings.ai.usd_inr:g}/US$)",
        f"Database: {settings.db_path}",
    ]
    print("\n".join(lines))
    return 0


def run(config_path: Path, dry_run: bool) -> int:
    load_dotenv(config_path.parent / ".env")          # before the Anthropic client is built
    settings = _load(config_path)
    if settings is None:
        return 2
    settings = settings_for_run(settings, dry_run)
    print(f"Database: {settings.db_path}"
          + (" (dry run: its own database; the paper-trading database is not touched)" if dry_run else ""))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"ANTHROPIC_API_KEY is not set in {config_path.parent / '.env'} or the environment.")
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from aitrader.app import LateBoundNotifier, build_services, schedule_jobs
    from aitrader.telegram_bot import TelegramNotifier, build_application

    notifier = LateBoundNotifier()
    svc = build_services(settings, notifier, dry_run)
    application = build_application(svc)
    notifier.target = TelegramNotifier(application.bot, settings.telegram_owner_chat_id)
    schedule_jobs(application, svc)
    logging.getLogger(__name__).info("aitrader starting%s; %d symbols in the universe",
                                     " in DRY-RUN mode (no orders)" if dry_run else "", len(svc.universe))
    try:
        application.run_polling()
    finally:
        svc.store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):           # ₹ on a Windows console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="aitrader", description="AI paper-trading bot for NSE (Phase 1)")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="start the bot and the schedule")
    run_p.add_argument("--dry-run", action="store_true", help="real data and AI, but place no orders at all")
    run_p.add_argument("--config", default="config.yaml")
    check_p = sub.add_parser("check-config", help="validate the config without connecting anywhere")
    check_p.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    if args.command == "check-config":
        return check_config(config_path)
    return run(config_path, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
