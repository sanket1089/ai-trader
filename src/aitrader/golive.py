"""Go-live report: the Phase 1 -> Phase 2 criteria of spec §9, checked against the store.

The report only informs: the owner makes the final decision. Every check is a
``(name, passed, detail)`` tuple, and the report is ``ready`` only when all of them pass.

State keys read: ``phase1_start`` (ISO date, set once by ``build_services``), ``hard_stop:<book>``
and ``stop_id:<book>`` (from ``LossLimits``), and the go-live flags ``tested_approval``,
``tested_stop`` and ``tested_expiry`` (set by ``control.py``).
"""

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from aitrader.calendar import IST
from aitrader.config import BookConfig, Settings
from aitrader.models import OrderStatus, Side
from aitrader.store import Store

MIN_WEEKS = 8
MIN_CLOSED_TRADES = 15
LEARNING, REALISTIC = "learning", "realistic"
TESTED_FLAGS = (("tested_approval", "an approval card was approved"),
                ("tested_stop", "/stop was used"),
                ("tested_expiry", "an approval card expired"))

Check = tuple[str, bool, str]


@dataclass(frozen=True)
class GoLiveReport:
    checks: list[Check]

    @property
    def ready(self) -> bool:
        return bool(self.checks) and all(ok for _, ok, _ in self.checks)


def _book(settings: Settings, name: str) -> BookConfig | None:
    return next((b for b in settings.books if b.name == name), None)


def _start(store: Store) -> date | None:
    value = store.get_state("phase1_start")
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _latest_equity(store: Store, book: str, today: date) -> float | None:
    """The most recent closing equity snapshot on or before ``today``."""
    return store.latest_equity_snapshot_before(book, today + timedelta(days=1))


def _months(start: date, end: date) -> list[tuple[int, int]]:
    out, (y, m) = [], (start.year, start.month)
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def evaluate_go_live(store: Store, settings: Settings, now: datetime, bench_return_pct: float) -> GoLiveReport:
    """Check every §9 criterion. ``bench_return_pct`` is NIFTYBEES' buy-and-hold return since the start
    (NaN when unknown, which fails the benchmark check)."""
    today = now.astimezone(IST).date()
    start = _start(store)
    checks: list[Check] = []

    # 1a. at least 8 weeks of paper trading
    if start is None:
        checks.append((f"{MIN_WEEKS} weeks of paper trading", False, "phase 1 start date not recorded"))
    else:
        days = (today - start).days
        checks.append((f"{MIN_WEEKS} weeks of paper trading", days >= MIN_WEEKS * 7,
                       f"{days} days since {start.isoformat()} (need {MIN_WEEKS * 7})"))

    # 1b. at least 15 closed trades in the learning book (a closing trade is a SELL fill)
    since = datetime.combine(start or today, time.min, tzinfo=IST)
    sells = [f for f in store.fills(LEARNING, since, now + timedelta(seconds=1)) if f.side is Side.SELL]
    checks.append((f"{MIN_CLOSED_TRADES} closed trades in the learning book",
                   len(sells) >= MIN_CLOSED_TRADES, f"{len(sells)} closed"))

    # 2. learning book's net-of-charges return beats buy-and-hold NIFTYBEES
    learning = _book(settings, LEARNING)
    equity = _latest_equity(store, LEARNING, today)
    if learning is None or equity is None or not math.isfinite(bench_return_pct):
        why = ("no learning book" if learning is None else "no equity snapshot yet" if equity is None
               else "NIFTYBEES benchmark unavailable")
        checks.append(("learning book beats NIFTYBEES", False, why))
    else:
        ret = (equity / learning.capital - 1) * 100
        checks.append(("learning book beats NIFTYBEES", ret > bench_return_pct,
                       f"learning {ret:+.2f}% net of charges vs NIFTYBEES {bench_return_pct:+.2f}%"))

    # 3. realistic book never hard stopped and above its capital after charges
    realistic = _book(settings, REALISTIC)
    stopped = (store.get_state(f"hard_stop:{REALISTIC}") is True
               or int(store.get_state(f"stop_id:{REALISTIC}", 0)) != 0)
    r_equity = _latest_equity(store, REALISTIC, today)
    if realistic is None or r_equity is None:
        checks.append(("realistic book never stopped and positive", False,
                       "no realistic book" if realistic is None else "no equity snapshot yet"))
    else:
        checks.append(("realistic book never stopped and positive",
                       not stopped and r_equity > realistic.capital,
                       f"{'was hard stopped' if stopped else 'never hard stopped'}; equity ₹{r_equity:,.2f} "
                       f"vs capital ₹{realistic.capital:,.2f}"))

    # 4. no order filled without a risk approval
    bypassed = [o.id for o in store.orders_with_status(OrderStatus.FILLED) if not store.risk_approved(o.id)]
    checks.append(("every filled order passed the risk manager", not bypassed,
                   f"{len(bypassed)} filled order(s) without a risk approval" if bypassed
                   else "all filled orders were risk approved"))

    # 5. approval flow, /stop and card expiry each tested live
    missing = [label for key, label in TESTED_FLAGS if store.get_state(key) is not True]
    checks.append(("approval, /stop and expiry tested live", not missing,
                   "not yet: " + "; ".join(missing) if missing else "all three seen"))

    # 6. AI spend within the monthly cap every month
    cap = settings.ai.monthly_budget_usd
    over = [(f"{y}-{m:02d}", spent) for y, m in _months(start or today, today)
            if (spent := store.ai_spend_usd(y, m)) > cap]
    checks.append(("AI spend within budget", not over,
                   "over the cap in " + ", ".join(f"{ym} (US${s:,.2f})" for ym, s in over) if over
                   else f"every month within US${cap:,.2f}"))
    return GoLiveReport(checks)


def format_report(report: GoLiveReport) -> str:
    head = ("Go-live check: READY (all criteria met; the decision is yours)" if report.ready
            else "Go-live check: not ready yet")
    return "\n".join([head] + [f"{'✅' if ok else '❌'} {name}: {detail}" for name, ok, detail in report.checks])
