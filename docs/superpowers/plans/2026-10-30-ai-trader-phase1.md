# AI Trader — Phase 1 (Paper Trading) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python service that runs a team of Claude agents to research NSE stocks and ETFs. It paper-trades two virtual books (₹1,000 and ₹1,00,000) under hard, code-enforced risk rules, and is controlled from a Telegram bot with an autopilot switch.

**Architecture:**
- **One asyncio process.** python-telegram-bot's `Application` runs the bot, and its `JobQueue` (APScheduler, IST timezone) runs the daily schedule.
- **Shared services.** Code-only components (market data, charges, paper broker, risk, execution, exits, approvals) and AI agents (thin wrappers over one `LLMClient`) are wired together in a `Services` bundle.
- **One persistence layer.** All state lives in SQLite through a single `Store`, so a restart loses nothing.

**Tech Stack:**
- Python ≥3.11 (machine has 3.14 via `py`)
- `anthropic` SDK with Pydantic structured outputs
- `python-telegram-bot[job-queue]` v21+
- `yfinance`, `pandas`, `feedparser`, `httpx`, `pyyaml`, `python-dotenv`
- `pytest`, `pytest-asyncio`

**Spec:** `docs/superpowers/specs/2026-10-30-ai-trader-design.md` (read it before starting; section numbers below refer to it).

## Global Constraints

**Scope and instruments**
- Phase 1 only. No real broker and no real orders. All trading goes through the `Broker` interface; the only implementation is `PaperBroker`.
- Delivery (CNC) only, NSE cash equities and ETFs only. No F&O, no margin, no short selling, no intraday products.
- Whole units only (`qty` is an `int` ≥ 1). There are no fractional shares.
- Limit orders only.

**Books**

| Book | Capital | Daily loss limit | Total loss limit | Max entries / day | Max sell days / week | Max position size |
|---|---|---|---|---|---|---|
| `realistic` | ₹1,000 | ₹100 | ₹500 | 1 | 3 | 100% of equity |
| `learning` | ₹1,00,000 | ₹10,000 | ₹50,000 | 3 | none | 20% of equity |

**Risk rules**
- Cost rule: expected move × position value ≥ **3×** round-trip charges.
- Entry band: limit price within ±0.5% of last price. Exit band: limit price 1% through the last price (marketable).
- Entry window: **09:20–15:10 IST**. Exits may be placed from **09:15**. NSE trading days only.
- Stale data: no order if the quote is older than *data delay (15 min) + 10 min*.
- At most **1 order per second**.
- Approval cards expire at **10 minutes or 15:10**, whichever is first. Re-issue a card if the price has moved more than **0.5%** at the moment of approval.
- Exits:
  - Check stop and target levels every **3 minutes**.
  - Re-price an unfilled exit after **2 minutes**, at most **3** times, then alert the owner.
  - Stop and target exits are never blocked by loss limits, the sell-day cap or the AI budget, and they don't count toward the sell-day cap.

**AI**
- Monthly cap **US$100**, shown in ₹ at the config rate (default 88).
- At ≥80% of the cap, force effort `low` and run intraday AI re-runs at most every 2 hours.
- At ≥100%, make no AI calls. Exits keep working.
- All agents default to model `claude-opus-5-5`. Default effort: news `low`, technical `low`, fundamental `low`, portfolio_manager `medium`, reviewer `medium`. Configurable per agent. **Never switch the model automatically.**
- Every call uses `betas=["server-side-fallback-2026-07-01"]` and `fallbacks="default"`. Check `stop_reason == "refusal"` before using output.
- Price table (USD per million tokens):

| Model | Input | Output | Cache read | Cache write |
|---|---|---|---|---|
| `claude-opus-5-5` | 4 | 20 | 0.20 | 5 |
| `claude-sonnet-5-5` | 2 | 10 | 0.20 | 2.5 |
| `claude-haiku-4-5` | 1 | 5 | 0.10 | 1.25 |

**Charges model (delivery)**

| Charge | Rate |
|---|---|
| Brokerage | 0 (configurable) |
| STT | 0.1% on buy and on sell |
| Stamp duty | 0.015% on buy |
| Exchange fee | 0.00322% |
| SEBI fee | 0.0001% |
| GST | 18% on (brokerage + exchange + SEBI) |
| DP charge | ₹15.34 per sell |

**Equity, limits and control**
- Equity = cash + Σ qty × price − Σ estimated sell charges for each holding (including the DP charge).
- Daily loss is measured from the previous trading day's closing equity snapshot. Total loss is measured from the baseline (starting capital, or the equity at the last approved recovery reset).
- Hitting the total limit means a hard stop: no new entries, autopilot set OFF, an alert, and one recovery plan per stop. Holdings are never force-sold.
- Autopilot defaults to **OFF**.
- Telegram answers only `TELEGRAM_OWNER_CHAT_ID`.

**Secrets and environment**
- Secrets live only in `.env`: `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_OWNER_CHAT_ID`. `.env` is gitignored. Never log or send secret values.
- Timezone is always `ZoneInfo("Asia/Kolkata")`. Never use the machine's local time.
- Shell is Windows. Run Python as `.venv/Scripts/python`. `python` is not on PATH, so bootstrap with `py -3`.

## Review Focus

1. **Process restarts mid-day.** Autopilot, pause and hard-stop state, pending approval cards, pending orders and positions must all survive a restart. Test in Task 2 (`test_state_survives_reopen`).
2. **yfinance returns empty, NaN or partial data for a symbol** (delisted, renamed or throttled). That symbol is skipped with a warning and is never priced at 0 or NaN. Tests in Task 3 (`test_quote_empty_frame_returns_none`, `test_quote_nan_returns_none`).
3. **The owner taps ✅ twice, or after expiry.** The second tap is a no-op, and a tap after expiry never places the trade. Tests in Task 11 (`test_double_approve_is_idempotent`, `test_approve_after_expiry`).
4. **The machine clock is not in IST, or a job fires on an NSE holiday.** Every decision uses IST, and holiday jobs do nothing. Tests in Task 2 (`test_now_ist_is_aware`) and Task 14 (`test_jobs_noop_on_holiday`).
5. **Not enough cash, or a sell for more than is held.** The paper broker rejects the order and never makes cash or quantity negative. Tests in Task 4 (`test_buy_rejected_insufficient_cash`, `test_sell_more_than_held_rejected`).

---

## File Structure

```
ai-trader/
  pyproject.toml            # deps, pytest config, package "aitrader" (src layout)
  config.yaml               # all non-secret settings (§ Global Constraints values)
  .env.example  .gitignore  README.md
  src/aitrader/
    __main__.py             # CLI: run [--dry-run] [--config], check-config
    config.py               # Settings (pydantic) + load_settings
    models.py               # domain dataclasses/enums
    calendar.py             # IST clock, trading days, windows
    charges.py              # charge model
    store.py                # SQLite Store
    data/market_data.py     # MarketData protocol, YFinanceMarketData, is_stale
    data/bhavcopy.py        # NSE EOD closes
    data/news.py            # RSS fetch + code pre-filter
    data/indicators.py      # IndicatorSnapshot + compute_indicators
    data/screener.py        # shortlist
    broker/base.py          # Broker protocol
    broker/paper.py         # PaperBroker
    risk/equity.py          # book_equity
    risk/limits.py          # LossLimits state machine
    risk/manager.py         # RiskManager.check
    execution.py            # Executor (risk gate, throttle, dry-run)
    exits.py                # ExitWatcher
    approvals.py            # ApprovalBook
    sizing.py               # size_idea per book
    agents/llm.py           # LLMClient + budget
    agents/schemas.py       # pydantic output models
    agents/analysts.py      # news / technical / fundamental agents
    agents/portfolio.py     # PM, recovery plan
    agents/reviewer.py      # daily + weekly review
    control.py              # command handlers as pure functions
    telegram_bot.py         # PTB adapter + TelegramNotifier
    pipeline.py             # premarket / open_session / intraday / exits / postmarket / weekly
    golive.py               # go-live criteria report
    app.py                  # Services wiring + job schedule
  tests/                    # one test file per module, same names: tests/test_<module>.py
```

---

### Task 1: Scaffold, config and charge model

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `config.yaml`, `src/aitrader/__init__.py`, `src/aitrader/config.py`, `src/aitrader/models.py`, `src/aitrader/charges.py`
- Test: `tests/test_charges.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `models.py`:
    - `Side(StrEnum)`: `BUY`, `SELL`
    - `OrderStatus(StrEnum)`: `PENDING`, `FILLED`, `CANCELLED`, `REJECTED`
    - `Quote(symbol: str, price: float, ts: datetime)`, frozen
    - `Order(id, book, symbol, side, qty: int, limit_price: float, reason: str, is_exit: bool, idea_id: str|None, created_at: datetime, status: OrderStatus = PENDING, attempts: int = 0)`
    - `Fill(order_id, book, symbol, side, qty, price, charges: float, is_exit: bool, ts)`
    - `Position(book, symbol, qty, avg_price, stop, target, opened_at, idea_id)`
    - `TradeIdea(id, symbol, side, limit_price, stop, target, expected_move_pct, confidence, holding_days, reason)`
  - `config.py`:
    - `ChargesConfig`
    - `BookConfig(name, capital, daily_loss_limit, total_loss_limit, max_entries_per_day, max_sell_days_per_week: int|None, max_position_pct)`
    - `RiskConfig(entry_band_pct, exit_band_pct, cost_multiple, data_delay_minutes, stale_extra_minutes, entry_start, entry_end, exit_start, max_orders_per_sec, exit_check_minutes, exit_reprice_minutes, exit_max_reprices)`
    - `ApprovalConfig(expiry_minutes, reprice_tolerance_pct)`
    - `AgentModel(model, effort)`
    - `ModelPrice(input, output, cache_read, cache_write)`
    - `AIConfig(monthly_budget_usd, usd_inr, reduced_at_pct=80, agents: dict[str, AgentModel], prices: dict[str, ModelPrice])`
    - `DataConfig(etfs: list[str], nifty100_csv_url, rss_feeds: list[str], shortlist_size)`
    - `ScheduleConfig` (HH:MM strings: `premarket`, `open_session`, `intraday` list, `postmarket`, `weekly_day`, `weekly_time`)
    - `Settings(books: list[BookConfig], charges, risk, approval, ai, data, schedule, holidays: list[date], db_path, telegram_owner_chat_id: int, telegram_bot_token: SecretStr)`
    - `load_settings(config_path: Path, env: Mapping[str,str] | None = None) -> Settings`
  - `charges.py`:
    - `ChargeBreakdown(brokerage, stt, exchange, sebi, stamp, gst, dp)` with property `total`
    - `compute_charges(side: Side, qty: int, price: float, cfg: ChargesConfig) -> ChargeBreakdown`
    - `round_trip_charges(qty: int, buy_price: float, sell_price: float, cfg) -> float`

- [ ] **Step 1: Bootstrap the environment**

Write `pyproject.toml`:
- src layout, package `aitrader`, `requires-python = ">=3.11"`
- deps: `anthropic`, `pydantic>=2`, `pyyaml`, `python-dotenv`, `yfinance`, `pandas`, `feedparser`, `httpx`, `python-telegram-bot[job-queue]>=21`
- `[project.optional-dependencies] dev = ["pytest", "pytest-asyncio"]`
- `[tool.pytest.ini_options] asyncio_mode = "auto"`

`.gitignore` covers `.venv/`, `.env`, `*.db`, `data/cache/`, `__pycache__/`. `.env.example` lists the three secret names with empty values.

Run: `py -3 -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"`
Expected: installs without error.

- [ ] **Step 2: Write `config.yaml`** with every value from Global Constraints:
- both books
- the charge rates
- the risk, approval and AI settings, including the price table and per-agent model and effort
- ETFs `NIFTYBEES, GOLDBEES, LIQUIDBEES, BANKBEES`
- `nifty100_csv_url: https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv`
- RSS feeds (Economic Times markets, Moneycontrol market reports, NSE announcements `https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml`)
- `shortlist_size: 25`
- the schedule: premarket 08:30, open_session 09:20, intraday 10:20/11:20/12:20/13:20/14:20, postmarket 15:45, weekly Saturday 10:00
- the 2026 and 2027 NSE trading holidays, copied from nseindia.com, with a comment giving the source URL and date checked

- [ ] **Step 3: Write failing tests**

```python
# tests/test_charges.py
CFG = ChargesConfig(brokerage_per_order=0.0, stt_pct=0.1, stamp_buy_pct=0.015,
                    exchange_pct=0.00322, sebi_pct=0.0001, gst_pct=18.0, dp_per_sell=15.34)

def test_buy_charges_niftybees():
    c = compute_charges(Side.BUY, 3, 280.0, CFG)          # turnover 840
    assert c.stt == pytest.approx(0.84)
    assert c.stamp == pytest.approx(0.126)
    assert c.dp == 0
    assert c.total == pytest.approx(0.99890784, abs=1e-6)

def test_sell_charges_include_dp():
    c = compute_charges(Side.SELL, 3, 300.0, CFG)         # turnover 900
    assert c.stamp == 0
    assert c.dp == 15.34
    assert c.total == pytest.approx(16.2752584, abs=1e-6)

def test_round_trip():
    assert round_trip_charges(3, 280.0, 300.0, CFG) == pytest.approx(0.99890784 + 16.2752584, abs=1e-6)

# tests/test_config.py
def test_load_settings_reads_yaml_and_env(tmp_path): ...
    # copies repo config.yaml; env={"TELEGRAM_OWNER_CHAT_ID": "42", "TELEGRAM_BOT_TOKEN": "x"}
    # asserts the realistic book has capital 1000, daily 100, total 500;
    # ai.monthly_budget_usd == 100; ai.agents["portfolio_manager"].model == "claude-opus-5-5"
def test_missing_owner_chat_id_raises(tmp_path): ...  # pydantic ValidationError
def test_secret_not_in_repr(tmp_path): ...            # "x" not in repr(settings)
```

- [ ] **Step 4: Run** `.venv/Scripts/python -m pytest tests/test_charges.py tests/test_config.py -v`. Expected: FAIL (import errors).
- [ ] **Step 5: Implement** `models.py`, `config.py` and `charges.py` per the Interfaces. In `load_settings`, read `.env` with `dotenv_values()` when `env` is None, then merge in `os.environ`.
- [ ] **Step 6: Run** the same tests. Expected: PASS.
- [ ] **Step 7: Commit** with `git add -A && git commit -m "feat: scaffold, config, charge model"`

---

### Task 2: IST calendar and SQLite store

**Files:**
- Create: `src/aitrader/calendar.py`, `src/aitrader/store.py`
- Test: `tests/test_calendar.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `models.py` (Task 1)
- Produces:
  - `calendar.py`:
    - `IST`
    - `now_ist() -> datetime`, timezone-aware
    - `is_trading_day(d: date, holidays: Iterable[date]) -> bool`
    - `previous_trading_day(d, holidays) -> date`
    - `at(d: date, hhmm: str) -> datetime`, aware IST
    - `in_window(t: datetime, start: str, end: str) -> bool`
  - `store.py` `Store(path: str | Path)`, using `sqlite3` with WAL. Methods:
    - **Orders and fills:** `save_order(Order)`, `update_order(Order)`, `pending_orders(book: str|None=None) -> list[Order]`, `save_fill(Fill)`, `fills(book, start: datetime, end: datetime) -> list[Fill]`
    - **Positions and cash:** `positions(book) -> list[Position]`, `upsert_position(Position)`, `remove_position(book, symbol)`, `cash(book) -> float`, `set_cash(book, value)`
    - **State:** `get_state(key, default=None) -> Any`, `set_state(key, value)` (JSON-encoded)
    - **Ideas and cards:** `save_idea(TradeIdea)`, `get_idea(id) -> TradeIdea|None`, `save_card(card_id, payload: dict)`, `get_card(card_id) -> dict|None`, `cards_with_status(status: str) -> list[dict]`
    - **Risk audit:** `save_risk_decision(order_id, approved: bool, reasons: list[str], ts)`, `risk_approved(order_id) -> bool`
    - **Agents and AI usage:** `save_agent_output(agent, payload: str, cost_usd: float, ts)`, `add_ai_usage(model, input_tokens, output_tokens, cache_read, cache_write, cost_usd, ts)`, `ai_spend_usd(year: int, month: int) -> float`, `ai_spend_usd_on(d: date) -> float`
    - **Journal:** `add_lesson(text, ts)`, `recent_lessons(n: int) -> list[str]`
    - **Equity:** `save_equity_snapshot(book, d: date, equity: float)`, `equity_snapshot(book, d) -> float|None`
    - **Counters:** `entries_on(book, d) -> int` (non-exit BUY fills), `bought_today(book, symbol, d) -> bool`, `sell_days_in_week(book, d) -> int` (distinct dates with non-exit SELL fills, Monday through `d`)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_calendar.py
def test_weekend_not_trading(): assert not is_trading_day(date(2026,10,31), [])          # Saturday
def test_holiday_not_trading(): assert not is_trading_day(date(2026,11,9), [date(2026,11,9)])
def test_previous_trading_day_skips_weekend(): assert previous_trading_day(date(2026,11,2), []) == date(2026,10,30)
def test_now_ist_is_aware(): assert now_ist().utcoffset() == timedelta(hours=5, minutes=30)
def test_in_window_bounds():
    assert in_window(at(date(2026,11,2),"09:20"), "09:20","15:10")
    assert not in_window(at(date(2026,11,2),"15:11"), "09:20","15:10")

# tests/test_store.py
def test_state_survives_reopen(tmp_path):
    s = Store(tmp_path/"t.db"); s.set_state("autopilot", True); s.save_order(order()); s.upsert_position(pos())
    s2 = Store(tmp_path/"t.db")
    assert s2.get_state("autopilot") is True
    assert len(s2.pending_orders()) == 1
    assert s2.positions("learning")[0].symbol == "NIFTYBEES"
def test_ai_spend_month_sums(tmp_path): ...      # two usages in Oct ($1.5, $2), one in Nov → Oct == 3.5
def test_sell_days_counts_distinct_non_exit_days(tmp_path): ...
    # Mon and Tue non-exit sells, Wed exit sell → 2
def test_entries_on_ignores_exits(tmp_path): ...
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_calendar.py tests/test_store.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `calendar.py` and `store.py`. Create the schema in `Store.__init__`: tables `orders`, `fills`, `positions`, `cash`, `state`, `ideas`, `cards`, `agent_outputs`, `ai_usage`, `lessons`, `equity`, `risk_log`. Store datetimes as ISO strings with offset.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "feat: IST calendar and SQLite store"`

---

### Task 3: Market data (yfinance, bhavcopy, staleness)

**Files:**
- Create: `src/aitrader/data/__init__.py`, `data/market_data.py`, `data/bhavcopy.py`
- Test: `tests/test_market_data.py`, `tests/test_bhavcopy.py`

**Interfaces:**
- Consumes: `Quote`, `RiskConfig`, `IST`
- Produces:
  - `MarketData(Protocol)`:
    - `quote(symbol) -> Quote|None`
    - `quotes(symbols) -> dict[str, Quote]` (missing symbols omitted)
    - `daily_history(symbol, days: int) -> pd.DataFrame` (columns `open, high, low, close, volume`; empty on failure)
    - `fundamentals(symbol) -> dict` (`{}` on failure)
    - `index_snapshot() -> dict` (keys `nifty_pct`, `banknifty_pct`, `fii_net_cr`, `dii_net_cr`; values may be `None`)
  - `YFinanceMarketData(yf_module=yfinance, http_get=httpx.get)`: maps `SYM` to `SYM.NS`, Nifty to `^NSEI` and Bank Nifty to `^NSEBANK`. For quotes it uses `yf.download(tickers, period="1d", interval="5m", group_by="ticker", progress=False)`; the quote ts is the last bar's timestamp converted to IST. Fundamentals come from `Ticker(...).info` keys `trailingPE, priceToBook, returnOnEquity, debtToEquity, marketCap, earningsGrowth, sector`. FII/DII come from `https://www.nseindia.com/api/fiidiiTradeReact`, wrapped in try/except so failures give `None`.
  - `is_stale(q: Quote, now: datetime, cfg: RiskConfig) -> bool`: true when `now - q.ts > data_delay_minutes + stale_extra_minutes`
  - `bhavcopy.fetch_closes(d: date, http_get=httpx.get) -> dict[str, float]`:
    - Downloads the NSE CM UDiFF zip `https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip` with a browser User-Agent.
    - Keeps rows with `SctySrs` in `{"EQ","BE"}` and maps `TckrSymb → ClsPric`.
    - Returns `{}` on any failure. The caller then falls back to the last yfinance daily close.

- [ ] **Step 1: Write failing tests** using a fake `yf_module`, a stub with `download`/`Ticker` returning prepared DataFrames, and a fake `http_get`:
  - `test_quote_maps_ns_suffix_and_ist_ts`
  - `test_quote_empty_frame_returns_none`
  - `test_quote_nan_returns_none`
  - `test_quotes_omits_missing_symbol`
  - `test_index_snapshot_fii_failure_gives_none`, where `http_get` raises and the result has `fii_net_cr is None`
  - `test_is_stale_boundary`: 25 minutes old is not stale, 25 minutes 1 second is stale
  - `test_bhavcopy_parses_eq_rows`: in-memory zip with EQ, BE and N1 rows; only EQ and BE are kept
  - `test_bhavcopy_failure_returns_empty`
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_market_data.py tests/test_bhavcopy.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules per the Interfaces.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "feat: market data via yfinance and NSE bhavcopy"`

---

### Task 4: PaperBroker

**Files:**
- Create: `src/aitrader/broker/__init__.py`, `broker/base.py`, `broker/paper.py`
- Test: `tests/test_paper_broker.py`

**Interfaces:**
- Consumes: `Store`, `MarketData`, `compute_charges`, `Order`, `Fill`, `Position`
- Produces:
  - `Broker(Protocol)`: `try_fill(order: Order, quote: Quote) -> Fill | None`, `cancel(order_id: str) -> None`
  - `PaperBroker(store, charges_cfg, slippage_pct: float = 0.30)`. The 0.30 is a base of 0.05 plus the extra 0.25 required by spec §7.3.

**Fill rules:**
- **BUY** fills at `quote.price × (1 + slippage%)` if that price ≤ limit. **SELL** fills at `quote.price × (1 − slippage%)` if that price ≥ limit. Otherwise it returns `None` and the order stays PENDING.
- **On a fill:** update cash (BUY: −(value + charges); SELL: +(value − charges)), then the position. A BUY creates or averages the position, keeping the stop and target from the idea. A SELL reduces it and removes it at 0. Mark the order FILLED and save the fill.
- **Rejections** (status REJECTED, returns `None`):
  - `qty < 1`
  - a BUY where cash < value + charges
  - a SELL where qty > held

The order must be saved first. Position stop and target for a BUY come from `store.get_idea(order.idea_id)`.

- [ ] **Step 1: Write failing tests**
  - `test_buy_fills_with_slippage_and_charges`: ₹1,000 cash, BUY 3 @ limit 282, quote 280 → fill price 280.84 and cash = 1000 − 842.52 − charges
  - `test_buy_not_filled_above_limit`: stays PENDING
  - `test_buy_rejected_insufficient_cash`: REJECTED, cash unchanged
  - `test_sell_more_than_held_rejected`
  - `test_sell_removes_position_and_charges_dp`
  - `test_buy_averages_existing_position`
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_paper_broker.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `broker/base.py` and `broker/paper.py`.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "feat: paper broker"`

---

### Task 5: Equity and loss-limit state machine

**Files:**
- Create: `src/aitrader/risk/__init__.py`, `risk/equity.py`, `risk/limits.py`
- Test: `tests/test_limits.py`

**Interfaces:**
- Consumes: `Store`, `BookConfig`, `compute_charges`
- Produces:
  - `book_equity(store, book: str, quotes: dict[str, Quote], charges_cfg) -> float`. It uses `avg_price` for a holding with no quote and logs a warning.
  - `LimitState(StrEnum)`: `ACTIVE`, `DAILY_PAUSED`, `HARD_STOPPED`
  - `LimitEval(state: LimitState, newly_triggered: bool, daily_loss: float, total_loss: float)`
  - `LossLimits(store, book_cfg, holidays)`:
    - `evaluate(equity: float, now: datetime) -> LimitEval`
    - `state(now) -> LimitState`
    - `baseline() -> float`. Store key `baseline:<book>`, initialised to capital.
    - `reset_baseline(equity: float) -> None`. Clears the hard stop and increments `stop_id:<book>`.
    - `stop_id() -> int`

**Rules:**
- **Daily reference:** `equity_snapshot(book, previous_trading_day(today))`, falling back to the baseline if there is none.
- **DAILY_PAUSED:** stored as `paused_on:<book>=<date>`. It applies only while `today == paused_on`.
- **HARD_STOPPED:** stored as `hard_stop:<book>=true`. It persists until `reset_baseline`.
- **Precedence:** HARD_STOPPED wins over DAILY_PAUSED.
- **`newly_triggered`:** true only on the transition into a state.

- [ ] **Step 1: Write failing tests**
  - `test_equity_subtracts_estimated_sell_charges`: cash 160, 3 × NIFTYBEES @ 280 → 160 + 840 − sell charges(3, 280)
  - `test_daily_pause_at_exact_limit`: loss of exactly 100 pauses; loss of 99.99 does not
  - `test_daily_pause_clears_next_trading_day`
  - `test_hard_stop_at_total_limit_and_persists`: re-evaluated with recovered equity, it still reports HARD_STOPPED
  - `test_hard_stop_precedence_over_daily`
  - `test_newly_triggered_only_once`
  - `test_reset_baseline_clears_stop_and_bumps_stop_id`
  - `test_overnight_gap_counts_toward_daily`: previous snapshot 1000, opening equity 880 → paused
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_limits.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "feat: equity and loss-limit state machine"`

---

### Task 6: RiskManager

**Files:**
- Create: `src/aitrader/risk/manager.py`
- Test: `tests/test_risk_manager.py`

**Interfaces:**
- Consumes: `Store`, `Settings`, `LossLimits`, `is_stale`, `in_window`, `is_trading_day`, `round_trip_charges`
- Produces:
  - `RiskDecision(approved: bool, reasons: list[str])`
  - `RiskManager(store, settings, limits: dict[str, LossLimits], universe: set[str])`
  - `check(order: Order, quote: Quote | None, now: datetime, idea: TradeIdea | None = None) -> RiskDecision`. It collects **all** failing reasons, not just the first.

**Rule table.** "Entry" means a BUY. "Exit" means `order.is_exit`. "Discretionary sell" means a SELL that is not an exit.

| Rule | Entry | Discretionary sell | Exit |
|---|---|---|---|
| Trading day | ✔ | ✔ | ✔ |
| Time window | `entry_start`–`entry_end` | `entry_start`–`entry_end` | `exit_start`–15:30 |
| Quote present and not stale | ✔ | ✔ | ✔ |
| Symbol in universe | ✔ | — | — |
| `qty ≥ 1`; SELL qty ≤ held | ✔ | ✔ | ✔ |
| Limit within band | entry band, ±0.5% | entry band | exit band: limit ≥ price × (1 − 1%) |
| Limit state ACTIVE | ✔ | — | — |
| `entries_on(book, today) < max_entries_per_day` | ✔ | — | — |
| Not `bought_today(book, symbol)` | ✔ | — | — |
| Position value ≤ `max_position_pct` × equity | ✔ | — | — |
| Cost rule: `idea.expected_move_pct/100 × qty × limit ≥ cost_multiple × round_trip_charges` | ✔ (requires `idea`) | — | — |
| Sell-day cap: if today is not already a sell day, `sell_days_in_week < cap` | — | ✔ | — |

- [ ] **Step 1: Write failing tests.** Use one test per row, named `test_<rule>_blocks_entry` and so on. Also include:
  - `test_exit_allowed_when_hard_stopped`
  - `test_exit_allowed_when_sell_cap_reached`
  - `test_exit_allowed_at_0915`
  - `test_entry_blocked_at_0915`
  - `test_collects_multiple_reasons`
  - `test_cost_rule_blocks_tiny_move`: 3 × NIFTYBEES @ 280 with 2% expected move gives 16.8 < 3 × ~17.3, so it is rejected
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_risk_manager.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `RiskManager.check`.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "feat: risk manager rules"`

---

### Task 7: LLMClient and AI budget

**Files:**
- Create: `src/aitrader/agents/__init__.py`, `agents/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Store`, `AIConfig`
- Produces:
  - `BudgetLevel(StrEnum)`: `NORMAL`, `REDUCED`, `EXHAUSTED`
  - `budget_level(store, ai_cfg, now) -> BudgetLevel`, comparing month spend against `reduced_at_pct` and 100% of the cap
  - `cost_usd(usage, price: ModelPrice) -> float` = `(input_tokens*in + output_tokens*out + cache_read_input_tokens*cr + cache_creation_input_tokens*cw)/1e6`. A `None` usage field counts as 0.
  - `LLMClient(store, ai_cfg, client: anthropic.Anthropic | None = None, clock=now_ist)`
  - `run(agent: str, system: str, user: str, output_model: type[T], max_tokens: int = 16000) -> T | None`

**`run` behaviour:**
- **Budget EXHAUSTED:** return `None` without calling the API.
- **Budget REDUCED:** use effort `low`.
- **Call shape:**

```python
resp = self.client.beta.messages.parse(
    model=m.model, max_tokens=max_tokens,
    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": user}],
    output_format=output_model,
    output_config={"effort": effort},
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
)
```

If the installed SDK's `beta.messages.parse` rejects `fallbacks` (`TypeError`), check the SDK's Python README under "Refusal Fallbacks" for the current spelling before changing it. Do not drop it silently.

- **After the call:**
  1. Record usage (`add_ai_usage`) and the agent output.
  2. If `resp.stop_reason == "refusal"`, log `stop_details` and return `None`.
  3. Otherwise return `resp.parsed_output`.
- **Errors:**
  - `anthropic.RateLimitError`, `APIConnectionError` and `APIStatusError` with status ≥ 500: log, then return `None`. The SDK has already retried twice.
  - `BadRequestError`: re-raise, because it is a bug.

- [ ] **Step 1: Write failing tests** with a fake client object exposing `beta.messages.parse` and returning a `SimpleNamespace` response:
  - `test_cost_usd_opus`: 10k in, 2k out, 50k cache read, 0 write → (10000×4 + 2000×20 + 50000×0.2)/1e6 = 0.09
  - `test_run_returns_parsed_and_records_usage`
  - `test_run_passes_fallbacks_and_effort`: asserts the kwargs `betas`, `fallbacks == "default"`, `output_config["effort"] == "low"` for the news agent, and `model == "claude-opus-5-5"`
  - `test_reduced_budget_forces_low_effort`: seed $80 spend; the PM call uses `low`
  - `test_exhausted_budget_skips_call`: seed $100; the fake is never called
  - `test_refusal_returns_none`
  - `test_rate_limit_returns_none`
  - `test_bad_request_raises`
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_llm.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `agents/llm.py`.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Live smoke test** (only if `ANTHROPIC_API_KEY` is set; it costs about $0.01). Run `.venv/Scripts/python -c "from aitrader.agents.llm import smoke; smoke()"`, where `smoke()` sends a tiny request with `output_model` `class Ok(BaseModel): ok: bool`. Expected: prints `ok=True` and a cost under $0.05. If the key is not set, skip this step and say so in the task report.
- [ ] **Step 6: Commit** with `git add -A && git commit -m "feat: LLM client with structured outputs, fallbacks and budget"`

---

### Task 8: News, indicators, screener

**Files:**
- Create: `data/news.py`, `data/indicators.py`, `data/screener.py`
- Test: `tests/test_news.py`, `tests/test_indicators.py`, `tests/test_screener.py`

**Interfaces:**
- Produces:
  - **`data/news.py`:**
    - `NewsItem(title, summary, link, published: datetime, source)`
    - `fetch_news(feeds: list[str], since: datetime, parse=feedparser.parse) -> list[NewsItem]`. Dedupes by link and drops items older than `since`. A failing feed is skipped with a warning.
    - `prefilter(items, symbols: dict[str, list[str]], market_keywords: list[str]) -> list[NewsItem]`. `symbols` maps each symbol to aliases, such as the company name from the Nifty 100 CSV. It keeps items that mention a symbol or alias, or a market keyword such as RBI, Sensex, Nifty, FII, inflation or crude. Matching is case-insensitive on word boundaries.
  - **`data/indicators.py`:**
    - `IndicatorSnapshot(symbol, close, sma20, sma50, rsi14, atr14, vol_ratio20, pct_change_5d, high_20d, low_20d)`
    - `compute_indicators(symbol, df: pd.DataFrame) -> IndicatorSnapshot | None`. Returns `None` with fewer than 50 rows. RSI uses Wilder smoothing. Plain pandas, no TA library.
  - **`data/screener.py`:**
    - `load_universe(csv_text: str, etfs: list[str]) -> tuple[list[str], dict[str, list[str]]]`. Parses the NSE Nifty 100 CSV columns `Symbol` and `Company Name` into the symbols and their aliases.
    - `screen(snaps: dict[str, IndicatorSnapshot], news_hits: dict[str, int], held: set[str], max_price: float | None, n: int) -> list[str]`. Score = |pct_change_5d| + 2×max(vol_ratio20 − 1, 0) + 3×news_hits. Held symbols are always included, plus the top-scoring symbols up to `n` in total. `max_price` excludes symbols priced above it (not used for held).

- [ ] **Step 1: Write failing tests**
  - `test_fetch_news_dedupes_and_filters_since`
  - `test_failing_feed_skipped`
  - `test_prefilter_matches_alias_case_insensitive`
  - `test_prefilter_word_boundary`: "TCS" does not match "TCSL"
  - `test_sma_on_linear_series`: closes 1..60 give sma20 == 50.5 and sma50 == 35.5
  - `test_rsi_monotonic_up_is_100`
  - `test_short_history_returns_none`
  - `test_screen_always_includes_held`
  - `test_screen_respects_n_and_max_price`
  - `test_load_universe_parses_csv`
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_news.py tests/test_indicators.py tests/test_screener.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "feat: news fetch, indicators, screener"`

---

### Task 9: Agent schemas and analyst agents

**Files:**
- Create: `agents/schemas.py`, `agents/analysts.py`
- Test: `tests/test_analysts.py`

**Interfaces:**
- Consumes: `LLMClient.run`, `NewsItem`, `IndicatorSnapshot`
- Produces, in `agents/schemas.py` (Pydantic `BaseModel`s):
  - `SymbolNews(symbol, sentiment: float  # -1..1, materiality: Literal["low","medium","high"], summary: str)`
  - `NewsDigest(market_mood: Literal["bullish","neutral","bearish"], market_summary: str, symbols: list[SymbolNews])`
  - `TechnicalView(symbol, trend: Literal["up","down","sideways"], support: float, resistance: float, signal: Literal["buy","hold","sell","avoid"], confidence: float, notes: str)`
  - `TechnicalBatch(views: list[TechnicalView])`
  - `FundamentalView(symbol, quality: Literal["strong","ok","weak"], valuation: Literal["cheap","fair","expensive"], notes: str)`
  - `FundamentalBatch(views: list[FundamentalView])`
  - `IdeaOut(symbol, side: Literal["BUY","SELL"], limit_price: float, stop: float, target: float, expected_move_pct: float, confidence: float, holding_days: int, reason: str)`
  - `PMDecision(market_view: str, ideas: list[IdeaOut])`
  - `RecoveryPlan(what_went_wrong: str, changes: list[str], mode: Literal["reduced_size","exit_only","normal"], next_trades: list[IdeaOut])`
  - `TradeReview(symbol, verdict: Literal["good","bad","neutral"], note: str)`
  - `ReviewOutput(lessons: list[str], trade_reviews: list[TradeReview])`
- Produces, in `agents/analysts.py`:
  - `run_news(llm, items: list[NewsItem], universe: list[str]) -> NewsDigest | None`. Agent key `"news"`.
  - `run_technical(llm, snaps: list[IndicatorSnapshot]) -> TechnicalBatch | None`. Key `"technical"`. One batched call for the shortlist.
  - `run_fundamental(llm, data: dict[str, dict]) -> FundamentalBatch | None`. Key `"fundamental"`. Weekly, in batches of 25.
- **Post-validation** (in code, not in the prompt): drop any symbol not in the provided inputs, clamp `sentiment` to [-1, 1] and `confidence` to [0, 1], and log what was dropped.
- **Prompts:**
  - Each prompt is a module-level constant system prompt, kept stable so it can be cached, plus a user message built from the data rendered as compact JSON.
  - The wording is the builder's call. Each prompt must state: Indian NSE cash market only, delivery swing trades of days to weeks, no F&O, use only the data given, and say "avoid" or "neutral" when the data is thin.

- [ ] **Step 1: Write failing tests** with a fake `LLMClient` whose `run` records its arguments and returns preset models:
  - `test_news_user_message_contains_titles_and_universe`
  - `test_news_drops_unknown_symbols`
  - `test_technical_clamps_confidence`
  - `test_fundamental_batches_of_25`: 60 symbols make 3 calls
  - `test_returns_none_when_llm_none`
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_analysts.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "feat: analyst agents and output schemas"`

---

### Task 10: Portfolio Manager, sizing, recovery plan, Reviewer

**Files:**
- Create: `agents/portfolio.py`, `agents/reviewer.py`, `src/aitrader/sizing.py`
- Test: `tests/test_portfolio.py`, `tests/test_sizing.py`, `tests/test_reviewer.py`

**Interfaces:**
- Consumes: the schemas (Task 9), `Store`, `BookConfig`, `round_trip_charges`, `TradeIdea`
- Produces:
  - **`agents/portfolio.py`:**
    - `PMContext(now, news: NewsDigest|None, technicals: TechnicalBatch|None, fundamentals: dict[str, FundamentalView], index: dict, positions: list[Position]  # learning book, cash: dict[str, float], lessons: list[str], quotes: dict[str, float], charges_note: str)`
    - `run_portfolio_manager(llm, ctx: PMContext) -> list[TradeIdea]`. Key `"portfolio_manager"`. It returns ideas with new uuid4 ids and saves each one with `store.save_idea`.
    - **Post-validation drops an idea when:**
      - the symbol has no quote
      - for a BUY, the order of stop < limit < target does not hold
      - `limit_price` is more than 0.5% from the quote
      - a SELL is for a symbol that is not held
    - Every drop is logged.
    - The prompt includes the charges note: "a sell costs about ₹15.34 DP plus 0.1% STT; tiny moves lose money".
    - `run_recovery_plan(llm, ctx: PMContext, stop_reason: str) -> RecoveryPlan | None`
  - **`sizing.py`:** `size_idea(idea: TradeIdea, book: BookConfig, cash: float, equity: float, held_qty: int) -> int`
    - **BUY:** `floor(min(cash, max_position_pct/100 × equity) / limit_price)`. Returns 0 if that is < 1.
    - **SELL:** `held_qty`.
  - **`agents/reviewer.py`:**
    - `run_daily_review(llm, store, day: date, books: list[str]) -> ReviewOutput | None`. Key `"reviewer"`. Its input is the day's fills, the ideas with their reasons, and the closing quotes. It appends each lesson with `store.add_lesson`.
    - `run_weekly_review(llm, store, week_end: date) -> str | None`. It returns plain-text weekly notes, using a `WeeklyNotes(notes: str)` schema.

- [ ] **Step 1: Write failing tests**
  - `test_size_realistic_takes_all_cash`: cash 1000, 100%, limit 280 → 3
  - `test_size_learning_caps_20pct`: equity 100000, limit 280 → floor(20000/280) = 71
  - `test_size_zero_when_unaffordable`
  - `test_pm_drops_bad_stop_order`
  - `test_pm_drops_far_limit`
  - `test_pm_drops_sell_not_held`
  - `test_pm_saves_ideas`
  - `test_pm_context_includes_lessons_and_charges_note`
  - `test_reviewer_appends_lessons`
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_portfolio.py tests/test_sizing.py tests/test_reviewer.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "feat: portfolio manager, sizing, reviewer"`

---

### Task 11: Approval cards and autopilot state

**Files:**
- Create: `src/aitrader/approvals.py`
- Test: `tests/test_approvals.py`

**Interfaces:**
- Consumes: `Store`, `ApprovalConfig`, `RiskConfig.entry_end`, `TradeIdea`
- Produces:
  - `get_autopilot(store) -> bool` (default `False`), `set_autopilot(store, on: bool)`
  - `ApprovalCard(id, idea_id, sizes: dict[str, int], quote_at_issue: float, expires_at: datetime, status: Literal["open","approved","rejected","expired"])`
  - `ApprovalBook(store, approval_cfg, entry_end: str)`:
    - `create(idea, sizes, quote, now) -> ApprovalCard`. `expires_at = min(now + expiry, at(today, entry_end))`.
    - `resolve(card_id, approve: bool, current_price: float, now) -> Resolution`, where `Resolution(StrEnum)` is `APPROVED`, `REJECTED`, `EXPIRED`, `ALREADY_RESOLVED`, `REISSUE`, `UNKNOWN`.
    - `REISSUE` is returned when approving and |current/quote_at_issue − 1| > tolerance. The old card becomes `expired`.
    - `expire_due(now) -> list[ApprovalCard]` marks overdue open cards as expired.
    - `render(card, idea) -> str` produces the card text: symbol, side, quantity per book, limit, the reason, target and stop, estimated charges per book, and the expiry time.

- [ ] **Step 1: Write failing tests**
  - `test_autopilot_defaults_off`
  - `test_expiry_is_min_of_10min_and_1510`: created at 15:05 → expires 15:10
  - `test_approve_within_tolerance`
  - `test_reissue_when_price_moved`: 0.6% move → REISSUE
  - `test_double_approve_is_idempotent`: second call → ALREADY_RESOLVED
  - `test_approve_after_expiry`: → EXPIRED
  - `test_unknown_card`
  - `test_render_contains_reason_and_sizes`
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_approvals.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `approvals.py`.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "feat: approval cards and autopilot state"`

---

### Task 12: Executor and ExitWatcher

**Files:**
- Create: `src/aitrader/execution.py`, `src/aitrader/exits.py`
- Test: `tests/test_execution.py`, `tests/test_exits.py`

**Interfaces:**
- Consumes: `RiskManager.check`, `Broker.try_fill`, `Store`, `MarketData.quotes`, `LossLimits`
- Produces:
  - `Notifier(Protocol)`: `async send(text: str) -> None`, `async send_card(card: ApprovalCard, text: str) -> None`
  - `Executor(store, broker, risk, notifier, dry_run: bool, min_interval_s: float = 1.0, sleep=asyncio.sleep, clock=time.monotonic)`:
    - `async submit(order, quote, now, idea=None) -> Fill | None`
    - Flow:
      1. Throttle to 1 per second.
      2. Run `risk.check` and record the result with `store.save_risk_decision`. A rejection sets the order to REJECTED, saves it, sends a notification with the reasons and returns `None`.
      3. In dry-run, save the order as CANCELLED with reason "dry-run", send a notification and return `None`.
      4. Otherwise save the order and call `broker.try_fill`. A fill sends a notification.
    - `async retry_pending(quotes, now) -> list[Fill]` retries PENDING non-exit orders and cancels entries left from earlier days.
  - `ExitWatcher(store, executor, market_data, risk_cfg, books: list[str])`: `async check(now) -> list[Fill]`
    - For each position with a quote, if `price ≤ stop` or `price ≥ target`, and the position has no pending exit, create an exit SELL with `is_exit=True` and `limit = round(price × (1 − exit_band), 2)`.
    - A pending exit that is ≥ `exit_reprice_minutes` old is re-priced (`attempts += 1`). After `exit_max_reprices` it is left pending, and the owner is notified once (state key `exit_alerted:<order_id>`).
    - Positions without a quote are skipped with a warning.

- [ ] **Step 1: Write failing tests**, using fakes for the broker, market data and notifier, and a fake sleep that records waits:
  - `test_rejected_order_notifies_reasons`
  - `test_dry_run_never_calls_broker`
  - `test_throttle_spaces_orders_1s`
  - `test_stop_hit_creates_exit`
  - `test_target_hit_creates_exit`
  - `test_no_duplicate_exit_when_pending`
  - `test_reprice_after_2min_max_3_then_alert_once`
  - `test_gap_down_open_exits_at_0915`
  - `test_exit_runs_when_hard_stopped`, using a real `RiskManager` with a hard-stopped book
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_execution.py tests/test_exits.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "feat: executor and exit watcher"`

---

### Task 13: Owner commands and Telegram adapter

**Files:**
- Create: `src/aitrader/control.py`, `src/aitrader/telegram_bot.py`
- Test: `tests/test_control.py`, `tests/test_telegram_bot.py`

**Interfaces:**
- Consumes: `Services` (defined in Task 14; tests use a fake with the same attributes: `store`, `settings`, `limits`, `approvals`, `executor`, `market_data`, `llm`), `get_autopilot` and `set_autopilot`, `LossLimits`, `budget_level`
- Produces, in `control.py`:
  - `handle_command(cmd: str, args: list[str], svc, now) -> str`. It covers `status`, `autopilot`, `positions`, `pnl`, `journal`, `pause`, `resume`, `stop`, `budget`, and `help` for unknown commands.
  - `async handle_card(card_id: str, approve: bool, svc, now) -> str`. Approving resolves the card. APPROVED submits the orders for each book through `svc.executor.submit` and returns a summary. REISSUE sends a fresh card.
  - `pending_recovery(store, book) -> dict|None` and `approve_recovery(store, book)`. These are used by the `recovery:<book>:<stop_id>` callback.
- **Command behaviour:**

| Command | Behaviour |
|---|---|
| `/status` | Autopilot, each book's limit state and headroom (₹), whether paused, AI budget level, last job run times |
| `/autopilot on\|off` | Sets the switch. `on` is refused while any book is HARD_STOPPED or `manual_stop` is set. |
| `/pause` | Sets `paused=true`, which blocks new entries for every book |
| `/stop` | Cancels all PENDING non-exit orders, sets `manual_stop=true` and autopilot off |
| `/resume` | Clears `paused` and `manual_stop`. For a HARD_STOPPED book, it resets that book's baseline to current equity only if `recovery_approved:<book>:<stop_id>` is true. The reply states the new baseline and that up to the total limit can be lost again from there. Otherwise it explains that a recovery plan must be approved first. |
| `/budget` | Month spend in USD and ₹, the percentage of the cap, and today's spend |

**Entry blocking.** `RiskManager` must also block entries when `paused` or `manual_stop` is set. Add both checks and their tests to `tests/test_risk_manager.py` in this task.

- Produces, in `telegram_bot.py`:
  - `build_application(svc) -> telegram.ext.Application`
    - Registers `CommandHandler`s with `filters.Chat(chat_id=owner)` and a `CallbackQueryHandler` for the patterns `^(approve|reject):` and `^recovery:`. It checks `query.from_user.id == owner` and edits the card message with the result.
    - A catch-all `MessageHandler(~filters.Chat(owner))` logs the chat id and does not reply.
  - `TelegramNotifier(bot, chat_id)` implements `Notifier`. `send_card` attaches an `InlineKeyboardMarkup` with ✅ `approve:<id>` and ❌ `reject:<id>`.

- [ ] **Step 1: Write failing tests**
  - `test_autopilot_on_refused_when_hard_stopped`
  - `test_stop_cancels_pending_entries_not_exits`
  - `test_resume_without_recovery_approval_refused`
  - `test_resume_after_recovery_resets_baseline_and_warns`
  - `test_pause_blocks_entries_in_risk`
  - `test_card_approve_submits_per_book`
  - `test_card_reissue_sends_new_card`
  - `test_budget_shows_usd_and_inr`
  - `test_non_owner_callback_ignored`: the PTB-level handler is called with a fake `Update` from another user id, and the executor is not called
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_control.py tests/test_telegram_bot.py tests/test_risk_manager.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `control.py`, `telegram_bot.py` and the two new risk checks.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "feat: owner commands and Telegram bot"`

---

### Task 14: Pipeline, go-live report, wiring and schedule

**Files:**
- Create: `src/aitrader/pipeline.py`, `src/aitrader/golive.py`, `src/aitrader/app.py`, `src/aitrader/__main__.py`, `README.md`
- Test: `tests/test_pipeline.py`, `tests/test_golive.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: everything above
- Produces, in `app.py`:
  - `Services` dataclass with `settings, store, market_data, broker, limits: dict[str, LossLimits], risk, executor, exits, approvals, llm, notifier, universe: list[str], aliases: dict[str, list[str]], dry_run: bool`
  - `build_services(settings, notifier, dry_run, market_data=None, llm_client=None) -> Services`
    - On first run, seeds cash and baseline for each book from its capital.
    - Downloads the Nifty 100 CSV and caches it at `data/cache/nifty100.csv`. If the download fails, it uses the cached copy. If there is no cache, it falls back to the ETFs only and warns.
  - `schedule_jobs(app: Application, svc) -> None`, using the JobQueue and IST times from `ScheduleConfig`:
    - `run_daily` for premarket, open_session, each intraday time and postmarket
    - `run_repeating` for exits every `exit_check_minutes`; the job itself returns outside 09:15–15:30
    - weekly on `weekly_day`
    - every job is wrapped by a guard that returns at once on non-trading days and catches and notifies exceptions
- Produces, in `pipeline.py` (all `async`, all taking `(svc, now)`):
  - **`premarket`:**
    - data health check: if no quote or daily history comes back for NIFTYBEES, send a "data feed down" alert and skip the AI steps
    - fetch news since the last run (state `news_since`) and prefilter it
    - `run_news`
    - daily history (60 rows) for the universe and `compute_indicators`
    - `screen`, with `max_price` = the realistic book's cash for that book's candidates, held symbols always included
    - `run_technical` on the shortlist
    - load fundamentals from state `fundamentals` (written weekly)
    - `run_portfolio_manager`
    - store the idea ids in state `todays_ideas`
  - **`open_session`:**
    - for each idea in `todays_ideas`, get a fresh quote and size it per book (skip books where the size is 0)
    - if autopilot is ON, `executor.submit` for each book
    - if OFF, create an approval card and `send_card`
    - skip entirely if `paused` or `manual_stop`
  - **`intraday`:**
    - `approvals.expire_due` and `executor.retry_pending`
    - re-run the PM only if the budget is not EXHAUSTED and a trigger fired: a held or shortlisted symbol moved more than 2% since premarket, or new high-materiality news. When the budget is REDUCED, run at most every 2 hours.
    - handle new ideas as in `open_session`
  - **`exits_job`:** `svc.exits.check`, then evaluate limits. When a total limit is newly triggered:
    - set autopilot OFF
    - send an alert
    - if the budget allows, call `run_recovery_plan` once per stop id (state `recovery_sent:<book>:<stop_id>`) and send it with a ✅ `recovery:<book>:<stop_id>` button
  - **`postmarket`:**
    - `fetch_closes(today)`, falling back to yfinance daily closes
    - for each book, `book_equity` at the closes, `save_equity_snapshot`, and `limits.evaluate`
    - `run_daily_review`
    - send the daily summary: per book, equity, P&L for the day, P&L since baseline, trades, charges paid, AI spend today and this month in USD and ₹, the NIFTYBEES benchmark since start (state `bench_start` is set on the first postmarket), and the lessons added
  - **`weekly`:** `run_fundamental` across the universe saved to state `fundamentals`, `run_weekly_review`, and the go-live report, all sent to Telegram
- Produces, in `golive.py`:
  - `GoLiveReport(checks: list[tuple[str, bool, str]])` with property `ready`
  - `evaluate_go_live(store, settings, now, bench_return_pct: float) -> GoLiveReport`, implementing spec §9:
    - at least 8 weeks since `phase1_start`
    - at least 15 closed trades in the learning book
    - the learning book's return net of charges beats `bench_return_pct`
    - the realistic book was never hard-stopped (`stop_id == 0` and not stopped) and its equity is above capital
    - no orders were filled without a risk approval: every FILLED order has `store.risk_approved(order_id)`
    - the state flags `tested_approval`, `tested_stop` and `tested_expiry` are true (set by `control.py` the first time each happens)
    - no month's AI spend went over the cap
- Produces, in `__main__.py`:
  - `py -m aitrader run [--dry-run] [--config config.yaml]` loads settings, builds the PTB app, notifier, services and schedule, then runs `app.run_polling()`
  - `py -m aitrader check-config` validates the config and prints the books, the schedule and the number of holidays, without connecting anywhere

- [ ] **Step 1: Write failing tests** with fakes for the market data, the `LLMClient` returning canned schema objects, and the notifier, and a real `Store` in `tmp_path`:
  - `test_jobs_noop_on_holiday`
  - `test_premarket_data_down_alerts_and_skips_ai`
  - `test_premarket_stores_todays_ideas`
  - `test_open_session_autopilot_off_sends_cards`
  - `test_open_session_autopilot_on_submits_both_books`
  - `test_open_session_skips_when_paused`
  - `test_realistic_book_skipped_when_unaffordable`
  - `test_exits_job_hard_stop_turns_autopilot_off_and_sends_one_recovery`: run twice and the recovery plan is sent once
  - `test_postmarket_saves_snapshots_and_summary_has_ai_spend_inr`
  - `test_postmarket_bhavcopy_fallback`
  - `test_intraday_skips_ai_when_exhausted`
  - `test_golive_not_ready_before_8_weeks`
  - `test_golive_ready_when_all_pass`
  - `test_build_services_seeds_capital_once`
  - `test_check_config_cli`: run through `subprocess` with the repo `config.yaml` and dummy env; exit code 0 and the output contains "realistic"
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_pipeline.py tests/test_golive.py tests/test_app.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `pipeline.py`, `golive.py`, `app.py` and `__main__.py`.
- [ ] **Step 4: Run** the full suite with `.venv/Scripts/python -m pytest -v`. Expected: all PASS.
- [ ] **Step 5: Write `README.md`** for the owner in plain language:
  - what the bot does and does not do
  - setup:
    1. create the bot with @BotFather
    2. get your chat id
    3. fill in `.env`
    4. set a spending limit on the Anthropic console
  - `py -m aitrader check-config`, then `run --dry-run` for a day, then `run`
  - every Telegram command
  - what the daily summary means
  - the go-live checklist
  - a "never paste passwords or OTPs anywhere" note
- [ ] **Step 6: Commit** with `git add -A && git commit -m "feat: pipeline, go-live report, app wiring and schedule"`

---

### Task 15: End-to-end dry run against real data

**Files:**
- Create: `tests/test_e2e_day.py`, `scripts/simulate_day.py`

**Interfaces:**
- Consumes: `build_services`, the `pipeline.*` functions

- [ ] **Step 1: Write `tests/test_e2e_day.py`.** It is fully offline: a fake market data with a scripted price path for NIFTYBEES and one Nifty stock, and a fake LLM that returns one BUY idea. It drives the jobs through a simulated trading day: premarket 08:30, open_session 09:20 with autopilot ON, exits every 3 minutes while the price walks up to the target, then postmarket. Assertions:
  - the learning and realistic books each have one entry fill and one exit fill
  - cash equals the start + P&L − the exact charges from `compute_charges`
  - an equity snapshot was saved
  - the summary was sent, and lessons were stored
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_e2e_day.py -v`. Expected: PASS once the earlier tasks are in. Fix the integration bugs it finds in the owning modules, each with a regression test there.
- [ ] **Step 3: Write `scripts/simulate_day.py`.** It runs premarket and open_session once against **real** yfinance data and the **real** Claude API, with `dry_run=True` and a `PrintNotifier` that prints Telegram messages to the console instead of sending them.
- [ ] **Step 4: Run it**, but only if `ANTHROPIC_API_KEY` is set and the owner has agreed to the cost (about $1–3). Use `.venv/Scripts/python scripts/simulate_day.py`. Expected: it prints the news digest, the shortlist, the PM ideas with reasons and the cards that would have been sent, ends without a traceback, and reports the AI cost. If the key is not set, skip this step and say so.
- [ ] **Step 5: Commit** with `git add -A && git commit -m "test: end-to-end simulated trading day and live dry-run script"`
