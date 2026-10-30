# AI Trader — Design

Date: 2026-10-30
Status: Draft, awaiting owner approval

## 1. Intent

The owner wants a team of AI agents that researches the Indian stock market and trades on their behalf. It starts with virtual money. Later it moves to a very small real account (₹500–1,000), and the goal is to grow that amount over time.

The owner is not a trading expert and is relying on the agents to do the reading and analysis faster than a person could. They want to stay in control through their phone. They can let the agents run on their own, or require approval for every trade.

We are honest about the limits:

- **Profit is not guaranteed.**
- **At ₹500–1,000, fixed charges dominate.** The DP charge of about ₹15 on every sell is about 3% of a ₹500 position.

So Phase 2 is mainly about **proving the system works safely with real money**. It is not about fast growth. Growth comes later, if the strategy proves itself and the owner chooses to add capital.

## 2. Goals

1. **The agent team researches the market.** It reads prices, news, NSE company announcements, foreign (FII) and domestic (DII) institutional flows, macro and global cues, and company fundamentals. It produces trade ideas with written reasoning.
2. **There are two phases.**
   - **Phase 1:** paper trading with virtual money on live NSE prices.
   - **Phase 2:** a real account with ₹500–1,000, entered only after Phase 1 passes its go-live criteria (§9).
3. **A Telegram bot is the owner's control panel.** It sends approvals, alerts, reports and commands.
4. **There is an autopilot switch.**
   - **ON:** the agents trade on their own, within the risk rules.
   - **OFF:** every trade idea is sent to Telegram with ✅ / ❌ buttons and is placed only if approved.
5. **Hard loss limits are enforced in code.** The agents can never override them.
6. **The system learns.** A Reviewer agent keeps a lesson journal, and later decisions draw on it.

## 3. Non-goals

- Futures and options. Margins are far too large for the account, and SEBI's FY25 study found 91% of individual F&O traders lost money.
- Intraday trading, scalping or high-frequency trading.
- Margin, leverage or short selling.
- Multiple users. SEBI also forbids sharing or selling a personal algo.
- A web dashboard or mobile app. A dashboard may come later.
- Fully unattended operation for days. SEBI requires a 2FA login every day.

## 4. Anti-goals

The project has failed, even if the code works, if any of these happen:

- The account is wiped out because the agents kept trading to win back losses.
- Any order reaches the broker without passing the Risk Manager.
- A trade is placed that the owner cannot understand, because every trade must come with a plain-language reason.
- Trading charges plus AI API costs quietly eat more than the gains, and nobody notices.
- Any SEBI rule is broken. These include the static IP, the order rate limit, daily 2FA, and personal use only.
- Credentials leak, through a commit, a log or a Telegram message.

## 5. Constraints (researched October 2026; re-check before going live)

**SEBI retail algo framework**
- Fully in force from 1 April 2026.
- DIY algos are allowed for the owner's own account.
- Every order must come from a static IP whitelisted with the broker.
- The limit is 10 orders per second per exchange. We will stay far below it.
- The broker attaches an exchange Algo ID to each order.
- A 2FA login is required every day.
- No primary SEBI source bans AI-made decisions. A blog's claim that AI strategies need pre-approval was **not confirmed**, so re-check this before Phase 2.

**Costs on delivery (CNC) trades**
- STT is 0.1% on buy and on sell.
- Stamp duty is 0.015% on the buy.
- Exchange fees are about 0.003%.
- GST is 18% on brokerage and fees.
- The **DP charge is about ₹15.34 per stock per day on which it is sold.**
- Brokerage depends on the broker. Dhan, Zerodha and Kotak Neo charge ₹0 for delivery.

**Other limits**
- India has no fractional shares, so only whole units can be bought. At ₹1,000, that means only a few units of a cheap ETF or stock.
- The Claude API costs money, so its spend must be capped and reported.

## 6. Decisions made with the owner

| Topic | Decision |
|---|---|
| Market | Indian equities (NSE) |
| Phase 1 | Virtual money first |
| Phase 2 | Real account with **₹1,000** starting capital, only after Phase 1 succeeds |
| Control channel | Telegram bot |
| Autopilot | A switch: ON trades alone, OFF asks for approval on each trade |
| Total loss limit | **₹500** (Phase 2). Hitting it stops all trading and turns autopilot OFF. |
| Recovery after a stop | The agents may propose a recovery plan on Telegram. It runs only if the owner approves it. |
| Daily loss limit | **₹100** (Phase 2). Hitting it pauses trading until the next trading day. The owner accepted this default. |
| Trading style | Swing trading: holding for days to weeks |
| What it trades | ETFs and large-cap stocks priced within the budget |
| Agent team | As in §7, accepted by the owner |
| AI running cost | A lean budget is accepted. Estimated at a few hundred ₹/month. |
| Project location | `C:\only\ai-trader`, its own git repository |
| AI cost vs trading results | **Counted separately.** The AI bill is a learning cost. Trading results and the go-live criteria are judged net of trading charges only. AI spend is shown next to P&L in every daily summary and in `/budget`, so it is never hidden. |

## 7. How it works

### 7.1 The team

The agents use AI for judgment. The rules and the order placement are plain, predictable code.

| Component | Kind | Job |
|---|---|---|
| Market Data | code | Live quotes and historical candles from the data feed, plus indices and FII/DII flows. Detects stale data. |
| News & Sentiment | AI agent | Reads Economic Times and Moneycontrol RSS, NSE corporate announcements, RBI news and global cues. Produces a short, structured view for each symbol and for the market as a whole. |
| Technical Analyst | AI agent, supplied with indicators computed in code | Trend, support and resistance, momentum, volume |
| Fundamental Analyst | AI agent | Results, valuation, sector strength. Runs less often (weekly, or when results come out). |
| Portfolio Manager | AI agent | Weighs the analysts' views, the current portfolio, recent lessons and costs. Proposes trades with a reason, a target, a stop level and an expected move. |
| Risk Manager | **code** | Applies every rule in §8. Can veto any trade. Nothing bypasses it. |
| Execution | **code** | Places limit orders through the `Broker` interface only after the Risk Manager approves, and approval is given if autopilot is OFF. |
| Reviewer | AI agent | After the close, reviews each trade and decision against what actually happened. Appends lessons to the journal and writes a weekly review. |
| Telegram Bot | code | The owner interface (§7.4) |

**Agent output.** Every agent returns structured JSON that follows a schema. Its outputs and reasoning are stored, so any trade can be explained afterwards.

**AI models** (builder's call, mine to confirm)
- Claude Sonnet 5.5 (`claude-sonnet-5-5`) for the Portfolio Manager, the analysts and the Reviewer.
- Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) for bulk news triage and summaries.

### 7.2 Daily rhythm (IST, NSE trading days only, using the NSE holiday calendar)

| Time | Step |
|---|---|
| ~08:30 | Data feed health check. No owner login in Phase 1. Phase 2 login handling is decided in the Phase 2 plan (§10.2). |
| 08:30–09:10 | Pre-market research: news, global cues, analyst passes, then the Portfolio Manager proposes ideas for the day. |
| 09:20–15:10 | A few scheduled checks (my call: about hourly), not continuous AI calls. Ideas are re-validated against live prices before placement. Stop and target levels are watched in code. |
| 15:10 onward | No new entries |
| After 15:30 | Reviewer pass, then the daily Telegram summary: trades, profit and loss (P&L), charges, AI spend, lessons. |
| Weekends | Weekly review, plus a refresh of the fundamental analysis |

### 7.3 Broker abstraction

All trading goes through one `Broker` interface with two implementations:

- **`PaperBroker` (Phase 1).** My call: we build our own simulator rather than run OpenAlgo, to avoid a separately hosted dependency.
  - It fills orders against live quotes, with a configurable slippage.
  - It applies the **full real charge model** (§5), including the DP charge.
  - It allows whole units only, delivery (CNC) only, and no margin.
- **`LiveBroker` (Phase 2).** An adapter for the owner's real broker, chosen at go-live time. It should be a zero-delivery-brokerage broker: Dhan, Zerodha's free personal API, or Kotak Neo.

**Market data in Phase 1 needs no login and no account** (owner decision: no daily login during Phase 1):
- yfinance (`.NS` tickers) for intraday and daily bars and quotes. They are delayed, often by about 15 minutes, and the library is unofficial and sometimes breaks.
- NSE end-of-day bhavcopy files are the authoritative daily closes, used for P&L marking and loss limits.
- `PaperBroker` fills at the delayed price plus **extra slippage** (my call: 0.25% on top of the base slippage), so delayed data can't make results look too good.
- All data access sits behind a `MarketData` interface, so a broker feed can replace yfinance in Phase 2 without touching agents or risk code.

**Two paper books run side by side in Phase 1** (my call):
- **Realistic book: ₹1,000.** Same capital, same ₹100 / ₹500 limits and same charges as Phase 2. It shows whether the plan works at real size.
- **Learning book: ₹1,00,000.** Limits scale in proportion: ₹10,000 per day and ₹50,000 total. It lets the strategy take enough trades to judge it statistically.

Both books get the same research and the same ideas, sized separately for each.

### 7.4 Telegram bot

- **It responds only to the owner's chat ID**, which is configured. Everything else is ignored and logged.
- **Commands:**
  - `/status`: mode, autopilot state, loss-limit headroom, last login
  - `/autopilot on|off`
  - `/positions`, `/pnl`, `/journal`
  - `/pause`, `/resume`
  - `/stop`: kill switch. It cancels open orders, blocks new ones and sets autopilot OFF.
  - `/budget`: AI spend
- **Approval cards** show the symbol, buy or sell, quantity, limit price, the reason in 2–3 lines, target and stop, estimated charges, and ✅ / ❌ buttons.
  - Unanswered cards expire after **10 minutes** (my call) or at 15:10, whichever comes first. An expired card is never placed.
  - Prices are re-checked when the owner approves. If the price has moved more than a set tolerance, the card is re-issued.
- **Alerts:** every fill, limit hits, stale data, failed login, errors, and the AI budget nearing its cap.

### 7.5 Storage, configuration and secrets

- **SQLite** holds orders, fills, positions, the daily P&L ledger, agent outputs, the lesson journal and AI usage.
- **`config.yaml`** holds limits, the universe, schedule, models and budget.
- **Secrets live in `.env` and are never committed** (`.gitignore` from the first commit). They include broker API keys, the Telegram bot token and the Anthropic API key.
- The owner never sends passwords, PINs or OTPs to the bot or to Claude.
- Logs mask secrets.

### 7.6 Hosting

- **Phase 1:** the owner's Windows PC, or a small VPS (builder's call).
- **Phase 2:** a small VPS in India with a **static IP** that is whitelisted with the broker, as SEBI requires. Expected cost is roughly ₹500–1,000 a month.

## 8. Risk rules (enforced in code by the Risk Manager)

**Loss limits** count realized plus unrealized P&L, net of all charges.

- **Daily loss limit**
  - Phase 2: ₹100. Realistic book: ₹100. Learning book: ₹10,000.
  - When hit, it blocks new entries until the next trading day and alerts the owner.
- **Total loss limit**
  - Phase 2: ₹500 below starting capital. Realistic book: ₹500. Learning book: ₹50,000.
  - When hit, it is a **hard stop**: it blocks all new entries, sets autopilot OFF and alerts the owner.
  - Positions are not force-sold by default. The owner decides through `/stop` or by approving an exit plan (my call; this avoids panic-selling at a low).
  - The Portfolio Manager may then send one **recovery plan**. Trading resumes only after the owner approves it **and** sends `/resume`.

**Instrument rules**
- Delivery (CNC) only, on NSE cash-market equities and ETFs from the configured universe.
- No F&O, no margin, no short selling, no intraday products.

**Order rules**
- **Limit orders only**, priced within a set band of the last traded price.
- **Cost check:** a trade's expected move must be at least 3× its estimated round-trip charges (my call).
- **Turnover caps** (my call), to stop charges from eating the account:
  - at most 1 new entry per day
  - at most 3 sell days per week in the ₹1,000 book or the live account
- A stock already held is never bought again on the same day.

**Freshness and timing**
- No order goes out if the data is stale. Stale means the feed has not updated for 10 minutes beyond its normal delay during market hours (my call).
- Orders are placed only between 09:20 and 15:10 IST, on NSE trading days.

**Rate and spend limits**
- A hard cap of 1 order per second, far below SEBI's 10 per second.
- **AI budget cap** per month, configurable, with a ₹500 default (my call).
  - At 80%, the system switches to fewer and cheaper calls.
  - At 100%, the AI agents stop and no new ideas are generated. Code still watches stop and target levels on existing positions.

### 8.1 Exits (my call)

**An approved entry comes with its exits approved.**
- When the owner approves a buy, or autopilot places one, the trade's stop and target levels are approved along with it.
- When a stop or target is hit, the code sells **automatically**, whether autopilot is ON or OFF.
- Other sells the Portfolio Manager proposes, such as closing early on news, follow the autopilot setting like entries do.

**Sells are never blocked by:**
- the daily or total loss limits
- the sell-day cap (stop and target exits don't count against it, and are always allowed)
- the AI budget cap

**How stops are watched:**
- Stop and target levels are checked **every 2–5 minutes** in code, depending on what yfinance can sustain, with no AI involved. Delayed data means stops can slip, so stops in Phase 1 are judged against what a real stop would have done.
- An exit is placed as a limit order inside a marketable band.
- If it isn't filled within 2 minutes, it is re-priced, up to N times. If it still hasn't filled, the owner is alerted.

**Stops that gap through:** if a stock opens past its stop, it is sold in the first check after 09:15 rather than waiting for the 09:20 entry window.

### 8.2 Measuring the loss limits (my call)

- **Equity** = cash + the market value of holdings − the estimated charges to sell them (including the DP charge).
- **Daily loss** is measured from the previous trading day's closing equity, so overnight gaps count.
- **Total loss** is measured from the baseline capital.

### 8.3 After a hard stop (my call)

**Recovery plan:** the Portfolio Manager sends one recovery plan per stop. It contains:
- what went wrong
- proposed changes, such as smaller size, a narrower universe or exit-only mode
- the next trades it would make

**Resuming:** if the owner approves the plan and sends `/resume`, the baseline resets to current equity and the limits start over from there.
- The confirmation message states this reset plainly, including how much more could be lost.

**If the owner rejects the plan:** the system stays stopped, and only exits are allowed.

### 8.4 Two books in Phase 1 (my call)

- **One Portfolio Manager run per cycle.** It sees the learning book's portfolio and produces a ranked idea list.
- **Code sizes each idea per book.** The ₹1,000 book takes the top idea it can afford, if any.
- **Autopilot and approvals also apply in Phase 1.** One approval card covers both books, so the flow is exercised before real money.
- **Turnover caps are set per book.**
  - Learning book: up to 3 new entries per day, with no sell-day cap.
  - ₹1,000 book: as in §8 (1 entry per day, 3 sell days per week).

## 9. Phase 1 to Phase 2: go-live criteria (my proposal)

All of these must hold before any real money is used. The owner makes the final decision.

1. At least **8 weeks** of paper trading, and at least **15 closed trades** in the learning book.
2. Over the period, the learning book's **net-of-charges return beats buying and holding NIFTYBEES**.
3. The realistic ₹1,000 book has not hit its total limit and has ended positive after charges.
4. There were zero risk-rule violations and zero orders that bypassed the Risk Manager.
5. The Telegram approval flow, `/stop` and the expiry of approval cards were each tested live at least once.
6. Total AI spend stayed within budget.

**Backtesting caveat.** AI agents cannot be honestly backtested on past data, because the model may already know what happened ("look-ahead bias"). Forward paper trading is the real test. Rule-based indicator code may still be backtested for sanity.

## 10. Testing

- **Unit tests** cover:
  - the charge model, checked against the broker's published charge calculator
  - every Risk Manager rule, including the edge cases at each limit
  - PaperBroker fills
  - Telegram owner filtering and approval expiry
- **Integration test:** the full pipeline runs end to end, with mocked AI and data responses.
- **A dry-run mode** runs the full day, with real data and real AI calls, but places no orders, not even paper ones.

## 10.1 Scope of the first build plan (my call)

The first plan builds **Phase 1 only**: the full agent team, `PaperBroker`, Telegram, risk and scheduling, all behind the `Broker` interface. `LiveBroker`, the VPS and static-IP setup come in a separate plan once the go-live criteria are met and the broker is chosen.

**AI budget sizing** (my call): to stay within the monthly cap, the AI-analysed universe starts at about 15–20 symbols, not the full Nifty 100. Intraday checks are code-only, and AI re-runs happen only when price moves or news trigger them.

## 10.2 Open items for the Phase 2 plan

- **Daily login.** SEBI requires a 2FA login every trading day before API orders can be placed. The owner declined a daily login in Phase 1, so Phase 2 must choose between:
  - a quick daily login tap that the owner accepts at that point
  - a broker whose API login uses a TOTP code that the program can generate securely, after checking this is allowed by the broker's terms and SEBI's rules at that time
- **Live data.** Move from delayed yfinance data to the chosen broker's live feed.
- **Broker, VPS and static IP.** Pick the broker, set up the VPS and whitelist its static IP.
- **AI-strategy approval.** Re-check the unconfirmed claim that AI strategies need exchange approval.

## 11. Left to the builder

- Exact libraries: for example `python-telegram-bot`, APScheduler, `pandas` / `pandas-ta`, and the Anthropic Python SDK.
- Project layout and module boundaries.
- Prompt wording and JSON schemas for each agent.
- The exact universe list (my starting suggestion: NIFTYBEES, GOLDBEES, LIQUIDBEES, plus Nifty 100 stocks whose price fits the budget).
- The slippage model, the price band percentage and the stale-data threshold.
- The source of the NSE holiday calendar.

## 12. What the owner will need to provide (when we get there; never in chat)

| When | What |
|---|---|
| Phase 1 | A Telegram account (we'll create the bot together with @BotFather) |
| Phase 1 | An Anthropic API key with a spending limit set |
| Phase 2 | A trading account at the chosen zero-brokerage broker, its API keys, and the VPS static IP whitelisted there |
