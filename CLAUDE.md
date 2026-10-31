# AI Trader: project guide for AI assistants

Read this first, then `docs/HANDOFF.md` for full status, decisions and backlog.

## What this is

A Python service where a team of Claude agents researches the Indian stock market (NSE cash equities and ETFs) and **paper-trades** two virtual books. The owner controls it from Telegram.

| | |
|---|---|
| Phase | **Phase 1, paper trading only**. There is no real broker and no real money. |
| Phase 2 | A real broker with ₹1,000. This needs its own design and plan; see `docs/HANDOFF.md` §6. |
| Owner | Not a trading expert. Prefers plain language, Telegram control, and safety over speed. |

## Key documents

| File | Contents |
|---|---|
| `docs/superpowers/specs/2026-10-30-ai-trader-design.md` | The approved design. This is the authority for intent. |
| `docs/superpowers/plans/2026-10-30-ai-trader-phase1.md` | The 15-task build plan. All tasks are done. Later rulings override parts of it; see HANDOFF. |
| `docs/HANDOFF.md` | Current status, every decision made during the build, the deferred backlog, and next steps. |
| `README.md` | Owner-facing setup and usage: Telegram commands, run order, go-live checklist. |
| `config.yaml` | All non-secret settings: books, limits, charges, schedule, holidays, AI budget and models. |

## Setup (new machine)

```bash
python -m venv .venv            # Windows: py -3 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Linux/macOS: .venv/bin/python
cp .env.example .env            # fill ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_OWNER_CHAT_ID
.venv/Scripts/python -m pytest -q              # 1246 tests, all offline
.venv/Scripts/python -m aitrader check-config
.venv/Scripts/python scripts/simulate_day.py   # first live dry run: real data and Claude, own DB, ~US$1-3
.venv/Scripts/python -m aitrader run           # the bot (paper trading)
```

## Architecture

- **One asyncio process.** python-telegram-bot's `Application` runs the owner's commands, and its JobQueue runs the IST schedule:
  - premarket at 08:30
  - open_session at 09:20
  - exits every 3 minutes from 09:15 to 15:30
  - intraday hourly
  - postmarket at 15:45
  - weekly on Saturday
- **SQLite** (`src/aitrader/store.py`) holds all state, so a restart loses nothing.
- **Code-only safety layer:**
  - `risk/manager.py` (`RiskManager`) is the single gate every order passes.
  - `risk/limits.py` latches the loss limits.
  - `execution.py` (`Executor`) is the only path to the broker.
  - `exits.py` holds the stop/target watcher.
  - `broker/paper.py` is `PaperBroker`, a simulated broker that applies full Indian charges plus slippage.
- **AI layer:** `agents/llm.py` (`LLMClient`) is the only path to Claude.
  - It uses the Anthropic SDK `beta.messages.create` with structured output and `fallbacks="default"`.
  - It enforces the monthly US$ budget and records every billed call.
  - The agents live in `agents/analysts.py`, `portfolio.py` and `reviewer.py`.
- **Data:**
  - `data/market_data.py` uses yfinance `.NS`, which is delayed about 15 minutes.
  - `data/bhavcopy.py` reads NSE closes.
  - `data/news.py` reads RSS feeds.
  - `data/indicators.py` and `data/screener.py` compute indicators and the shortlist.
- **Wiring:**
  - `app.py` holds `Services`, the schedule and the job guard.
  - `pipeline.py` and `pipeline_support.py` hold the job bodies.
  - `control.py` holds the owner commands and card handling.
  - `telegram_bot.py` is the PTB adapter.
  - `golive.py` checks the Phase 2 criteria.

## Non-negotiable rules

1. **Every order goes through `Executor` and then `RiskManager.check`.** Never call the broker directly.
2. **Stop and target exits are never blocked** by loss limits, `/pause`, the sell-day cap or the AI budget.
3. **Loss limits:** realistic book ₹100 per day and ₹500 total; learning book ₹10,000 per day and ₹50,000 total.
   - Hitting the total limit is a hard stop: autopilot goes OFF.
   - Trading resumes only after the owner approves a recovery plan and sends `/resume`.
4. **Autopilot defaults to OFF.** Telegram answers only the owner, checked by chat id **and** user id.
5. **Secrets live only in `.env`**, which is gitignored. Never log or send them. Errors are logged as type plus file:line only.
6. **Time:** use IST (`ZoneInfo("Asia/Kolkata")`) everywhere. Never use machine-local time.
7. **AI models:**
   - Never switch models automatically.
   - Model and effort are set per agent in `config.yaml`; the default is `claude-opus-5-5`.
   - Downgrading to a cheaper model is the owner's decision.
8. **Instruments:** delivery (CNC) only, whole units, limit orders. No F&O, no margin, no short selling.

## Conventions

- **TDD with pytest.**
  - Tests are fully offline, using fakes for market data, the LLM and the notifier, and a real `Store` in `tmp_path`.
  - `asyncio_mode = auto`.
- **Small focused modules.** Follow the existing style and comment density.
- **Commits** are small, with `feat:`, `fix:`, `test:` or `docs:` prefixes.
- **State keys in the store `state` table** are part of the contract between modules. Grep before renaming one.

## Environment notes

- The machine where this was built was on a corporate network.
  - That network blocked yfinance fundamentals, and Telegram needed the OS certificate store, which is why `truststore` is used at startup.
  - On a normal home network neither issue applies.
- Python 3.11 or newer. It was built and tested on 3.14.
