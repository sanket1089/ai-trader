"""RiskManager: the gate every order must pass. Nothing bypasses it.

``check`` collects every failing rule, not just the first, so the owner sees the whole picture.
Reasons are short, human-readable sentences (they are shown on Telegram).

Three kinds of order, with different rules:

* entry: a BUY. All rules apply.
* discretionary sell: a SELL that is not an exit. Same window and band as an entry, plus the
  weekly sell-day cap. Not blocked by the loss limits.
* exit: ``is_exit`` SELL (stop, target, forced). Wider window (``exit_start`` to the 15:30 close),
  a marketable-limit floor instead of a band, and never blocked by loss limits, the sell-day cap,
  the entry caps or the cost rule.

A BUY flagged ``is_exit`` is not an exit: it is rejected, and still judged by every entry rule.

Orders that are approved but not yet filled (PENDING in the store) already claim capacity, so the
counters, the held quantity and the position cap include them. The order under check is always
left out (matched by id), because the executor re-checks a pending order on retry and it must
not be counted against itself.

Money comparisons use ``TOLERANCE`` (a millionth of a rupee) so float noise at an exact boundary
cannot flip a decision; it is far below one tick, so it never lets a real breach through.
"""

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from aitrader.calendar import IST, at, in_window, is_trading_day
from aitrader.charges import round_trip_charges
from aitrader.config import BookConfig, Settings
from aitrader.data.market_data import is_stale
from aitrader.models import Order, Position, Quote, Side, TradeIdea
from aitrader.risk.equity import book_equity
from aitrader.risk.limits import TOLERANCE, LimitState, LossLimits
from aitrader.store import Store

MARKET_CLOSE = "15:30"   # exits may be placed until the close; entries stop at ``risk.entry_end``

_STATE_REASONS = {
    LimitState.DAILY_PAUSED: "daily loss limit reached: no new entries today",
    LimitState.HARD_STOPPED: "total loss limit reached: hard stop, no new entries",
}


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reasons: list[str]


def _usable(x: float) -> bool:
    return isinstance(x, int | float) and math.isfinite(x) and x > 0


def _aware(t: datetime) -> bool:
    return t.tzinfo is not None and t.utcoffset() is not None


def _created_on(o: Order, day: date) -> bool:
    """True if ``o`` was created on the IST calendar day ``day``."""
    return o.created_at.astimezone(IST).date() == day


class RiskManager:
    def __init__(self, store: Store, settings: Settings, limits: dict[str, LossLimits],
                 universe: set[str]):
        self._store = store
        self._settings = settings
        self._books = {b.name: b for b in settings.books}
        self._limits = dict(limits)
        self._universe = frozenset(universe)

    def check(self, order: Order, quote: Quote | None, now: datetime,
              idea: TradeIdea | None = None) -> RiskDecision:
        """Approve or reject ``order`` as of ``now`` (timezone-aware).

        ``quote`` is the latest quote for ``order.symbol`` (``None`` if there is none) and ``idea``
        the trade idea behind an entry (needed for the cost rule). Never raises for a bad order,
        quote or book: those are rejection reasons. Writes nothing.
        """
        if not _aware(now):
            raise ValueError("now must be timezone-aware")
        today = now.astimezone(IST).date()
        risk = self._settings.risk

        if order.side == Side.BUY:
            kind = "entry"
        elif order.side == Side.SELL:
            kind = "exit" if order.is_exit else "sell"
        else:
            return RiskDecision(False, [f"unknown order side {order.side!r}"])

        reasons: list[str] = []
        book_cfg = self._books.get(order.book)
        book_limits = self._limits.get(order.book)
        known = book_cfg is not None and book_limits is not None
        if not known:
            reasons.append(f"unknown book {order.book!r}")
        if kind == "entry" and order.is_exit:
            reasons.append("a BUY cannot be an exit")

        # Trading day and time window.
        if not is_trading_day(today, self._settings.holidays):
            reasons.append("not an NSE trading day")
        if kind == "exit":
            start, end, label = risk.exit_start, MARKET_CLOSE, "exit"
        else:
            start, end, label = risk.entry_start, risk.entry_end, "entry"
        if not in_window(now, start, end):
            reasons.append(f"outside {label} window {start}–{end}")

        # Quote.
        price, quote_reasons = self._check_quote(order, quote, now)
        reasons += quote_reasons

        if kind == "entry" and order.symbol not in self._universe:
            reasons.append(f"{order.symbol} is not in the trading universe")

        # Quantity. ``qty_ok`` also guards the rules below that do arithmetic on it.
        qty_ok = isinstance(order.qty, int) and order.qty >= 1
        if not qty_ok:
            reasons.append("quantity must be a whole number of at least 1")
        position = self._position(order) if known else None
        # Approved-but-unfilled orders of this book, without the order being checked.
        pending = self._pending(order) if known else []
        if kind != "entry" and qty_ok and known:
            held = position.qty if position is not None else 0
            pending_sells = sum(p.qty for p in pending
                                if p.side == Side.SELL and p.symbol == order.symbol)
            if order.qty > held - pending_sells:
                if pending_sells:
                    reasons.append(f"sell quantity {order.qty} exceeds the {held} held, of which "
                                   f"{pending_sells} already pending sale")
                else:
                    reasons.append(f"sell quantity {order.qty} exceeds the {held} held")

        # Limit price against the last price.
        limit_ok = _usable(order.limit_price)
        if not limit_ok:
            reasons.append("limit price must be a positive number")
        elif price is not None:
            reasons += self._check_band(kind, order.limit_price, price)

        if kind == "entry":
            if known:
                reasons += self._check_entry_book_rules(order, book_cfg, book_limits, now, today, pending)
            if known and qty_ok and limit_ok:
                reasons += self._check_position_value(order, book_cfg, position, quote, price, pending)
            if qty_ok and limit_ok:
                reasons += self._check_cost(order, idea)
        elif kind == "sell" and known:
            reasons += self._check_sell_days(order.book, book_cfg, today, pending)

        return RiskDecision(approved=not reasons, reasons=reasons)

    # ---- rules ----

    def _check_quote(self, order: Order, quote: Quote | None, now: datetime) -> tuple[float | None, list[str]]:
        """Reasons the quote cannot be used, and its price if it is a usable one for this symbol.

        A stale quote still has a usable price (the band is judged against it, and the staleness
        is reported as its own reason).
        """
        if quote is None:
            return None, ["no quote available"]
        if quote.symbol != order.symbol:
            return None, [f"quote is for {quote.symbol}, not {order.symbol}"]
        reasons = []
        price = quote.price if _usable(quote.price) else None
        if price is None:
            reasons.append("quote has no usable price")
        if not _aware(quote.ts):
            reasons.append("quote time has no timezone")
        elif is_stale(quote, now, self._settings.risk):
            minutes = self._settings.risk.data_delay_minutes + self._settings.risk.stale_extra_minutes
            reasons.append(f"quote is stale (older than {minutes} min)")
        return price, reasons

    def _check_band(self, kind: str, limit: float, price: float) -> list[str]:
        risk = self._settings.risk
        if kind == "exit":
            # Marketable: a sell limit at most ``exit_band_pct`` below the last price.
            floor = price * (1 - risk.exit_band_pct / 100)
            if limit < floor - TOLERANCE:
                return [f"exit limit {limit:.2f} is more than {risk.exit_band_pct:g}% below "
                        f"last price {price:.2f}"]
            return []
        width = price * risk.entry_band_pct / 100
        if abs(limit - price) > width + TOLERANCE:
            return [f"limit {limit:.2f} is outside ±{risk.entry_band_pct:g}% of last price {price:.2f}"]
        return []

    def _check_entry_book_rules(self, order: Order, book_cfg: BookConfig, book_limits: LossLimits,
                                now: datetime, today: date, pending: list[Order]) -> list[str]:
        """Limit state, entries per day and one-buy-per-symbol-per-day, counting pending buys."""
        reasons = []
        state = book_limits.state(now)
        if state is not LimitState.ACTIVE:
            reasons.append(_STATE_REASONS[state])
        pending_buys_today = [p for p in pending
                              if p.side == Side.BUY and not p.is_exit and _created_on(p, today)]
        entries = self._store.entries_on(order.book, today) + len(pending_buys_today)
        if entries >= book_cfg.max_entries_per_day:
            reasons.append(f"daily entry limit reached ({entries}/{book_cfg.max_entries_per_day})")
        if (self._store.bought_today(order.book, order.symbol, today)
                or any(p.symbol == order.symbol for p in pending_buys_today)):
            reasons.append(f"already bought {order.symbol} today")
        return reasons

    def _check_position_value(self, order: Order, book_cfg: BookConfig, position: Position | None,
                              quote: Quote | None, price: float | None,
                              pending: list[Order]) -> list[str]:
        """Value after the buy <= cap % of book equity.

        The value is this order plus the existing holding of the symbol plus its pending buys.
        Equity is marked at ``quote``; any other holding falls back to its average price.
        """
        quotes = {quote.symbol: quote} if quote is not None else {}
        try:
            equity = book_equity(self._store, order.book, quotes, self._settings.charges)
        except KeyError:
            return [f"cash has not been set for book {order.book!r}"]
        value = order.qty * order.limit_price
        if position is not None:
            # The same mark book_equity uses for this holding.
            value += position.qty * (price if price is not None else position.avg_price)
        value += sum(p.qty * p.limit_price for p in pending
                     if p.side == Side.BUY and p.symbol == order.symbol)
        cap = book_cfg.max_position_pct / 100 * equity
        if value > cap + TOLERANCE:
            return [f"position ₹{value:,.2f} would exceed {book_cfg.max_position_pct:g}% of equity "
                    f"(₹{cap:,.2f})"]
        return []

    def _check_cost(self, order: Order, idea: TradeIdea | None) -> list[str]:
        """Expected gain must cover ``cost_multiple`` times the round-trip charges."""
        if idea is None:
            return ["entry needs a trade idea for the cost check"]
        if idea.symbol != order.symbol:
            return ["idea is for a different symbol"]
        multiple = self._settings.risk.cost_multiple
        charges = round_trip_charges(order.qty, order.limit_price, order.limit_price,
                                     self._settings.charges)
        expected = idea.expected_move_pct / 100 * order.qty * order.limit_price
        if not expected >= multiple * charges - TOLERANCE:      # not-greater-equal: NaN fails
            return [f"expected gain ₹{expected:,.2f} is below {multiple:g}× round-trip charges "
                    f"₹{charges:,.2f}"]
        return []

    def _check_sell_days(self, book: str, book_cfg: BookConfig, today: date,
                         pending: list[Order]) -> list[str]:
        """A discretionary sell may start a new sell day only while this week has fewer than the cap.

        Today already counts as a sell day if a non-exit SELL filled today or one is pending that
        was created today (it was approved against the cap and will use today's slot).
        """
        cap = book_cfg.max_sell_days_per_week
        if cap is None:
            return []
        if any(p.side == Side.SELL and not p.is_exit and _created_on(p, today) for p in pending):
            return []
        day_fills = self._store.fills(book, at(today, "00:00"), at(today + timedelta(days=1), "00:00"))
        if any(f.side == Side.SELL and not f.is_exit for f in day_fills):
            return []                                           # today already counts as a sell day
        days = self._store.sell_days_in_week(book, today)
        if days >= cap:
            return [f"weekly sell-day cap reached ({days}/{cap} days)"]
        return []

    def _pending(self, order: Order) -> list[Order]:
        """PENDING orders in ``order.book``, excluding ``order`` itself (an executor retry re-checks it)."""
        return [p for p in self._store.pending_orders(order.book) if p.id != order.id]

    def _position(self, order: Order) -> Position | None:
        return next((p for p in self._store.positions(order.book) if p.symbol == order.symbol), None)
