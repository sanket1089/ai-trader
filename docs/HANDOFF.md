# Handoff: AI Trader (as of 2026-10-31)

## 1. Status

**Phase 1 (paper trading) is complete.**
- All 15 plan tasks are implemented, and each one passed a spec-and-quality review.
- A final whole-branch review found no Critical issues. Its Important findings were fixed and re-reviewed.
- 1246 offline tests pass.

**Never run live.** The app has not yet run against real Claude, Telegram or market hours. The first live step is `scripts/simulate_day.py` (see README), followed by a dry-run day.

The work moved from an office laptop to the owner's personal PC. All further work happens on the personal PC.

## 2. What the owner decided (brainstorming, 2026-10-30)

- **Market:** Indian NSE cash market. Swing trades over days to weeks, in ETFs and large caps.
- **Phases:**
  - Phase 1 is virtual money.
  - Phase 2 is a real account starting with **₹1,000**, only after the go-live criteria pass (spec §9).
- **Control:** a Telegram bot.
  - Autopilot ON means the agents trade alone.
  - Autopilot OFF means every trade comes as a ✅/❌ approval card.
- **Loss limits:** ₹100 per day and ₹500 total on the real-size book.
  - A total-limit hit stops all trading.
  - The agents may only *propose* a recovery plan; the owner approves it.
- **AI budget:** a cap of **US$100 per month**. The AI cost is counted **separately** from trading P&L.
- **No daily broker login in Phase 1.** So the data is delayed yfinance, not a live broker feed.
- **The agent team was accepted:** News, Technical, Fundamental, Portfolio Manager, Risk Manager (code), Execution (code) and Reviewer.

## 3. Key build decisions (controller rulings)

These override the plan text where they differ.

**Data and storage**
- **Atomic store writes.** `Store.transaction()` is re-entrant, and a paper fill is written atomically.
- **Daily-loss reference.** It is the previous trading day's equity snapshot. Failing that, it is the latest earlier snapshot, and only then the baseline.
- **Pending-aware risk.** PENDING orders count toward the entry caps, `bought_today`, held quantity (no oversell) and the position cap. The order under check is excluded.

**AI**
- **LLM calls.** They use `beta.messages.create` with an `output_config` JSON-schema format. Usage is **recorded before** validation, so refused, cut-off and invalid replies are still billed against the budget.
- **Pricing.** Each call is priced by the model that served it, with fallback model price rows in config.
- **PM safeguards.**
  - Expected move is capped at the limit-to-target distance (no bypassing the 3× cost rule).
  - Sizing nets out buy charges.
- **Prompt-injection hardening.** News and analyst text is marked untrusted in every prompt.

**Exits and stops**
- **Stop/target of 0.0 means "unset".** It never triggers an exit, and the owner is alerted once.
- **Exit re-pricing.** It works on the same order, every 2 minutes or more, with no attempt cap. The owner is alerted once past 3 attempts.
- **Before an exit.** Any pending discretionary SELL is cancelled first.
- **Per-position isolation.** An error on one position never stops the others.
- **Blind-stop alert.** It fires once per day per position, only after 09:40 (allowing for the feed delay).

**Entries and ideas**
- **Re-anchoring at the open.** Morning ideas are re-anchored to the fresh quote using `marketable_limit` (the top of the ±0.5% band for a BUY, so the paper fill at +0.30% slippage works).
  - Stop and target are kept.
  - An idea is skipped if the stop isn't clear of the fill price.
- **Stale or missing quote.** The idea waits, unacted. `exits_job` retries it without AI until fresh quotes arrive, usually around 09:40.
- **Entry de-duplication.** (symbol, side) is de-duplicated per day. The intraday trigger re-bases after each re-run.
- **Pause and stop.** `/pause` and `/stop` close today's open ideas. Ideas older than 2 hours are skipped.
- **Recovery plan.** It is sent once per stop. A plain-text ✅ fallback is offered when the budget is exhausted, so a stop is never permanent.

**Telegram and controls**
- **Owner-only.** Commands need both chat id and user id. A group owner id is refused. `/stop` never fakes a cancel and expires all open cards.
- **Cards.** A card left open ("tap again") keeps its buttons. Cards expire at 10 minutes or at 15:10.
- **`/resume`.** It fetches quotes off the event loop.

**Running modes and network**
- **Dry run.** `run --dry-run` uses its own `data/dryrun.db` and never seeds `phase1_start` or the `tested_*` go-live flags.
- **Corporate networks.** `truststore` is injected at startup, so HTTPS works behind TLS-inspecting proxies.

## 4. Backlog (deferred minors: none block paper trading)

**Before ~20 December 2026**
- Add the **2027 NSE holidays** to `config.yaml`. NSE hadn't published them yet.

**Watch in the first live week**
- Is yfinance data delayed about 15 minutes? Is `stale_extra_minutes` tuned right?
- Prompt-cache hit rate (`cache_read_input_tokens`) and actual daily AI spend.
- Do `/status` "last runs" show missed exits jobs?
- Morning "data feed down" false alarms.
- Do `resp.model` aliases fragment the AI ledger?

**Nice to have**
- Recovery plan: don't cache a `None` plan until one hourly retry has been made.
- Call `retry_pending` from `exits_job` inside the entry window.
- `notify_once` race and unguarded store calls in the error handlers.
- State-key cleanup: `exit_*`, `*_alerted:*` and `job_error:*` keys accumulate.
- `send_card` failure marks an idea acted.
- Re-issued cards keep their old limit and sizes. The limit is re-set at approval.
- Lessons are timestamped with the real clock, not the job's `now`.
- Too-old ideas linger in the waiting list (INFO log noise).
- Simulator and dry-run AI spend is not counted in the real DB's budget.
- Test-strength nits across modules: some loose assertions, a timing-sensitive `/resume` test, a partly vacuous CLI DB test.

## 5. Next steps on the personal PC

1. Clone the repo. Set up `.venv`, then run `pytest -q` (expect 1246 passing).
2. Create the Telegram bot with @BotFather and get your chat id. Create an Anthropic API key and **set a monthly spend limit in the Anthropic console**. Fill in `.env`.
3. Run `check-config`, then `scripts/simulate_day.py` on a trading day between about 09:30 and 15:10 IST.
4. Optionally run `run --dry-run` for a day, then `run`. The 8-week paper clock (`phase1_start`) starts on the first real run.
5. Keep the PC on 08:30–15:45 IST on trading days, or move to a small always-on machine or VPS.

## 6. Phase 2 (real money): not started; needs its own design and plan

- **SEBI retail algo framework**, in force from 1 April 2026:
  - orders must come from a static IP whitelisted with the broker;
  - a daily 2FA login is required (the owner declined a daily login for Phase 1, so revisit it);
  - at most 10 orders per second;
  - an exchange Algo ID per order.
- **Broker.** Pick a zero-delivery-brokerage broker (Dhan, Zerodha's personal API or Kotak Neo) and implement `LiveBroker` behind the `Broker` interface.
- **Data.** Switch to the broker's live feed.
- **Check the claim** that AI strategies need exchange approval. It was unconfirmed in research.
