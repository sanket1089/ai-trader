# AI trader (Phase 1: paper trading)

A small team of Claude agents researches NSE stocks every trading day and proposes trades. In
Phase 1 **no real money is involved**: every trade goes to a paper broker that simulates fills,
charges and slippage. You talk to the bot on Telegram.

## What it does

- **Before the market opens (08:30 IST)** it checks that market data is working, reads the news,
  computes indicators for the Nifty 100 stocks plus a few ETFs, shortlists about 25, and asks the AI
  team for trade ideas.
- **At 09:20** it sizes each idea for two paper books:
  - **realistic**: ₹1,000, the same size you would start with in real life;
  - **learning**: ₹1,00,000, so there are enough trades to judge the strategy.

  With autopilot **OFF** (the default) you get one approval card per idea with ✅ / ❌ buttons. With
  autopilot **ON** the ideas are placed straight away, still through every risk check.
- **Every hour (10:20 to 14:20)** it retries waiting orders and closes unanswered cards. It asks the AI
  again only if something moved more than 2% or important news came in.
- **Every 3 minutes (09:15 to 15:30)** it checks each holding's stop and target and sells
  automatically when one is hit. This happens whatever the autopilot setting is.
- **After the close (15:45)** it marks the books at the day's closing prices, has the AI review the
  day and write lessons, and sends you the daily summary. NSE often publishes its official closing
  prices (the bhavcopy) hours after 15:45, so the evening summary usually uses Yahoo's daily close
  instead.
- **Saturday 10:00**: the weekly review and the go-live checklist.

## What it does not do

- It never trades real money and never connects to a real broker in Phase 1.
- Delivery (CNC) only: no F&O, no margin, no short selling, no intraday trades. Limit orders only.
- It never sells your holdings because a loss limit was hit. Stops and targets still work.
- It never switches the AI model by itself, and it stops calling the AI when the monthly budget is used up.

## Setup

1. **Create the bot.** In Telegram, open **@BotFather**, send `/newbot` and follow the steps. Keep the
   token it gives you (it looks like `123456789:AA...`).
2. **Get your chat id.** Send any message to your new bot, then open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and look for
   `"chat":{"id":123456789`. That number is your chat id. (Or message **@userinfobot**.)
3. **Fill in `.env`.** Copy `.env.example` to `.env` in this folder and fill in:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   TELEGRAM_BOT_TOKEN=123456789:AA...
   TELEGRAM_OWNER_CHAT_ID=123456789
   ```
   `.env` is never committed to git. The bot answers only this chat id.
4. **Set a spending limit on the Anthropic console** (console.anthropic.com → Billing → Limits). The
   app has its own US$100/month cap, but a limit in the console as well is strongly recommended.

Install once:

```
py -3 -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
```

## Running it

1. Check the configuration (connects to nothing):
   ```
   .venv\Scripts\python -m aitrader check-config
   ```
   It prints the books, the schedule and how many holidays are configured.
2. See one morning without starting the bot: run the simulator. It does the 08:30 premarket and the
   09:20 open session **once** with real data and the real Claude API, places no orders and sends
   nothing to Telegram, and prints the news digest, the shortlist, the AI's ideas with their reasons, the
   approval cards you would have received, and what the run cost in AI spend (about US$1-3, so it needs
   `ANTHROPIC_API_KEY` in `.env`; without it the script says so and stops):
   ```
   .venv\Scripts\python scripts\simulate_day.py
   ```
   Cards appear only while quotes are fresh: run it on a trading day between about 09:30 and 15:10 IST.
   It uses its own database, `data/simulate.db` (recreated on every run), never the real one.
3. Optional: run the whole bot for a day in **dry-run** mode. It uses real data, real AI calls and
   Telegram, but places no orders at all, not even paper ones:
   ```
   .venv\Scripts\python -m aitrader run --dry-run
   ```
   A dry run uses its own database, `data/dryrun.db`, so it never starts the Phase 1 clock, ticks a
   go-live box or writes into the real paper-trading database. The database in use is printed at startup.
4. Then run it for real (paper trading):
   ```
   .venv\Scripts\python -m aitrader run
   ```
   Keep the window open. Stop it with Ctrl+C. It is safe to restart: cash, positions and limits are
   kept in `data/aitrader.db`.

**Every December:** add next year's NSE trading holidays to the `holidays` list in `config.yaml`
(see "Good to know" below).

## Telegram commands

| Command | What it does |
|---|---|
| `/status` | Autopilot, pause/stop switches, loss-limit headroom per book, AI budget, when each job last ran |
| `/autopilot on` / `/autopilot off` | Place ideas automatically, or ask you first with a card (default OFF) |
| `/positions` | Holdings and waiting orders per book |
| `/pnl` | Equity and profit/loss per book, and AI spend this month |
| `/journal [n]` | The latest lessons the reviewer wrote (default 5, up to 20) |
| `/pause` | No new entries and no AI sell ideas in any book; today's ideas not acted on yet are dropped. Automatic stop/target exits still work. |
| `/resume` | Lift a pause or stop. A hard-stopped book restarts only after you approve its recovery plan. Ideas dropped by `/pause` or `/stop` stay dropped. |
| `/stop` | Kill switch: cancels waiting entry orders, closes open approval cards, drops today's ideas not acted on yet, blocks new entries and turns autopilot off |
| `/budget` | AI spend this month and today, in US$ and ₹ |
| `/help` | The list of commands |

**Approval cards** show the stock, buy or sell, units per book, the limit price, target, stop, the
estimated charges and the reason. They expire after 10 minutes or at 15:10, whichever is first. If
the price moved more than 0.5% by the time you tap ✅, you get a fresh card instead. If a buy's
price has fallen to its stop by then, nothing is placed and the card is closed. An idea is acted on
for at most 2 hours after the AI made it.

**Loss limits.** The realistic book pauses for the day after a ₹100 loss and stops completely after
₹500; the learning book after ₹10,000 / ₹50,000. After a complete (hard) stop the AI sends one
recovery plan with a ✅ button. Approve it and send `/resume` to restart. The losses are then counted
again from the current equity.

## The daily summary

After the close you get, for each book:

- **equity**: cash plus holdings at the closing price, minus what it would cost to sell them
  (charges included);
- **today**: the change since yesterday's close;
- **since baseline**: the change since the starting capital (or since the last approved recovery);
- **trades** and **charges paid** today.

Then:

- **AI spend** today and this month, in US$ and in ₹ (at ₹88 per US$, set in `config.yaml`). It is a
  learning cost and is kept apart from trading P&L.
- **NIFTYBEES since the start**: what simply buying and holding the Nifty ETF would have made. This
  is the bar to beat.
- **Lessons added** by the reviewer today.

## Go-live checklist (every Saturday)

Real money is considered only when **all** of these hold. The final decision is always yours.

1. At least 8 weeks of paper trading.
2. At least 15 closed trades in the learning book.
3. The learning book's return after charges beats buying and holding NIFTYBEES.
4. The realistic ₹1,000 book was never hard stopped and is above ₹1,000 after charges.
5. Every filled order went through the risk manager.
6. You have tested an approval, `/stop` and a card expiring, at least once each.
7. No month's AI spend went over the cap.

## Good to know

- **Holidays:** update the `holidays` list in `config.yaml` every December when NSE publishes the next
  year's list. The 2027 list is still a TODO; until it is added, 2027 holidays would be treated as
  trading days.
- **Fundamentals:** on this corporate network the yfinance fundamentals are blocked, so the weekly
  message will usually say "fundamentals unavailable this week". The rest works without them.
- **Delayed data:** market data is delayed by about 15 minutes. "No fresh price" alerts are not sent
  before about 09:40, when the delayed feed has caught up with the open.
- **Logs** show only error types and places, never secrets.

## Safety

**Never paste passwords, PINs or OTPs anywhere**: not into Telegram, not into Claude, not into
`config.yaml`. The bot never needs them. The only secrets are the three lines in `.env`.
