import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from aitrader.calendar import at
from aitrader.charges import round_trip_charges
from aitrader.config import load_settings
from aitrader.models import Fill, Order, OrderStatus, Position, Quote, Side, TradeIdea
from aitrader.risk.limits import LimitState, LossLimits
from aitrader.risk.manager import RiskDecision, RiskManager
from aitrader.store import Store

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
SETTINGS = load_settings(REPO_CONFIG, env={"TELEGRAM_OWNER_CHAT_ID": "42", "TELEGRAM_BOT_TOKEN": "x"})
CHARGES = SETTINGS.charges
HOLIDAYS = SETTINGS.holidays

BOOK = "realistic"        # capital 1000, 1 entry/day, 3 sell days/week, position cap 100%
LEARNING = "learning"     # capital 100000, 3 entries/day, no sell-day cap, position cap 20%
SYMBOL = "NIFTYBEES"
UNIVERSE = {"NIFTYBEES", "GOLDBEES", "RELIANCE"}

FRI = date(2026, 10, 30)
MON = date(2026, 11, 2)
TUE, WED, THU = (MON + timedelta(days=i) for i in (1, 2, 3))
NEXT_MON = MON + timedelta(days=7)
SUN_HOLIDAY = date(2026, 11, 8)       # Sunday, also listed as a holiday
WEEKDAY_HOLIDAY = date(2026, 11, 10)  # Tuesday, Diwali-Balipratipada
NOON = at(MON, "12:00")

KINDS = ["entry", "sell", "exit"]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    s.set_cash(BOOK, 1000.0)
    s.set_cash(LEARNING, 100000.0)
    yield s
    s.close()


@pytest.fixture
def limits(store):
    books = {b.name: b for b in SETTINGS.books}
    return {name: LossLimits(store, books[name], HOLIDAYS) for name in (BOOK, LEARNING)}


@pytest.fixture
def rm(store, limits):
    return RiskManager(store, SETTINGS, limits, UNIVERSE)


# ---------------------------------------------------------------- helpers


def order(side=Side.BUY, qty=3, limit=280.0, is_exit=False, symbol=SYMBOL, book=BOOK, id="o1",
          created_at=NOON, status=OrderStatus.PENDING):
    return Order(id=id, book=book, symbol=symbol, side=side, qty=qty, limit_price=limit,
                 reason="test", is_exit=is_exit, idea_id="i1", created_at=created_at, status=status)


def buy(**kw):
    return order(Side.BUY, **kw)


def sell(**kw):
    return order(Side.SELL, **kw)


def exit_(**kw):
    return order(Side.SELL, is_exit=True, **kw)


def make(kind, **kw):
    return {"entry": buy, "sell": sell, "exit": exit_}[kind](**kw)


def quote(now=NOON, price=280.0, symbol=SYMBOL, age_minutes=1.0):
    return Quote(symbol=symbol, price=price, ts=now - timedelta(minutes=age_minutes))


def idea(move_pct=8.0, symbol=SYMBOL):
    return TradeIdea(id="i1", symbol=symbol, side=Side.BUY, limit_price=280.0, stop=270.0, target=300.0,
                     expected_move_pct=move_pct, confidence=0.7, holding_days=5, reason="trend")


GOOD_IDEA = idea()


def hold(store, symbol=SYMBOL, qty=3, avg_price=280.0, book=BOOK):
    store.upsert_position(Position(book=book, symbol=symbol, qty=qty, avg_price=avg_price, stop=270.0,
                                   target=300.0, opened_at=at(FRI, "11:00")))


def prep(store, kind, **kw):
    """Sells and exits need something to sell."""
    if kind != "entry":
        hold(store, **kw)


_fill_ids = iter(range(10_000))


def add_fill(store, side, ts, is_exit=False, symbol=SYMBOL, book=BOOK, qty=1, price=280.0):
    store.save_fill(Fill(order_id=f"f{next(_fill_ids)}", book=book, symbol=symbol, side=side, qty=qty,
                         price=price, charges=0.0, is_exit=is_exit, ts=ts))


def add_pending(store, side, qty=3, limit=280.0, is_exit=False, symbol=SYMBOL, book=BOOK,
                created=at(MON, "11:00"), status=OrderStatus.PENDING):
    """Save an approved-but-unfilled order (a different id from the order under check)."""
    o = order(side, qty=qty, limit=limit, is_exit=is_exit, symbol=symbol, book=book,
              id=f"p{next(_fill_ids)}", created_at=created, status=status)
    store.save_order(o)
    return o


def check(rm, o, now=NOON, q="fresh", i=GOOD_IDEA):
    """Run ``rm.check`` with a fresh matching quote (override ``q``, e.g. with None) and a good idea."""
    if isinstance(q, str):
        q = quote(now, symbol=o.symbol)
    return rm.check(o, q, now, i)


def only_reason(decision, fragment):
    """The order is rejected for exactly one reason, and it mentions ``fragment``."""
    assert not decision.approved
    assert len(decision.reasons) == 1, decision.reasons
    assert fragment in decision.reasons[0], decision.reasons


def has_reason(decision, fragment):
    assert not decision.approved
    assert any(fragment in r for r in decision.reasons), decision.reasons


# ---------------------------------------------------------------- baseline


def test_decision_shape(rm):
    d = check(rm, buy())
    assert isinstance(d, RiskDecision)
    assert d.approved is True
    assert d.reasons == []


@pytest.mark.parametrize("kind", KINDS)
def test_clean_order_is_approved(rm, store, kind):
    prep(store, kind)
    d = check(rm, make(kind))
    assert d.approved, d.reasons


def test_check_needs_aware_now(rm):
    with pytest.raises(ValueError, match="timezone-aware"):
        rm.check(buy(), quote(), datetime(2026, 11, 2, 12, 0), GOOD_IDEA)


def test_now_in_other_timezone_uses_ist_clock(rm):
    now = NOON.astimezone(timezone.utc)          # 06:30 UTC is 12:00 IST
    assert check(rm, buy(), now=now).approved


# ---------------------------------------------------------------- trading day


@pytest.mark.parametrize("day", [SUN_HOLIDAY, WEEKDAY_HOLIDAY, date(2026, 11, 7)])
def test_trading_day_blocks_entry(rm, day):
    only_reason(check(rm, buy(), now=at(day, "12:00")), "trading day")


@pytest.mark.parametrize("kind", ["sell", "exit"])
def test_trading_day_blocks_sell_and_exit(rm, store, kind):
    prep(store, kind)
    only_reason(check(rm, make(kind), now=at(WEEKDAY_HOLIDAY, "12:00")), "trading day")


# ---------------------------------------------------------------- time window


@pytest.mark.parametrize("hhmm", ["09:15", "09:19", "15:11", "15:30"])
def test_time_window_blocks_entry(rm, hhmm):
    only_reason(check(rm, buy(), now=at(MON, hhmm)), "outside entry window 09:20–15:10")


@pytest.mark.parametrize("hhmm", ["09:20", "15:10"])
def test_entry_window_edges_are_inclusive(rm, hhmm):
    assert check(rm, buy(), now=at(MON, hhmm)).approved


@pytest.mark.parametrize("hhmm", ["09:15", "15:11"])
def test_time_window_blocks_discretionary_sell(rm, store, hhmm):
    hold(store)
    only_reason(check(rm, sell(), now=at(MON, hhmm)), "outside entry window 09:20–15:10")


@pytest.mark.parametrize("hhmm", ["09:14", "15:31"])
def test_time_window_blocks_exit(rm, store, hhmm):
    hold(store)
    only_reason(check(rm, exit_(), now=at(MON, hhmm)), "outside exit window 09:15–15:30")


@pytest.mark.parametrize("hhmm", ["09:15", "15:15", "15:30"])
def test_exit_window_is_0915_to_1530(rm, store, hhmm):
    hold(store)
    assert check(rm, exit_(), now=at(MON, hhmm)).approved


def test_exit_allowed_at_0915(rm, store):
    hold(store)
    d = check(rm, exit_(), now=at(MON, "09:15"))
    assert d.approved, d.reasons


def test_entry_blocked_at_0915(rm):
    d = check(rm, buy(), now=at(MON, "09:15"))
    only_reason(d, "outside entry window 09:20–15:10")


# ---------------------------------------------------------------- quote present and fresh


@pytest.mark.parametrize("kind", KINDS)
def test_missing_quote_blocks_every_order_type(rm, store, kind):
    prep(store, kind)
    d = check(rm, make(kind), q=None)
    assert not d.approved
    assert "no quote" in " ".join(d.reasons)


@pytest.mark.parametrize("kind", KINDS)
def test_quote_for_another_symbol_blocks_every_order_type(rm, store, kind):
    prep(store, kind)
    other = quote(NOON, symbol="GOLDBEES")
    d = check(rm, make(kind), q=other)
    assert not d.approved
    assert any("GOLDBEES" in r and SYMBOL in r for r in d.reasons), d.reasons


@pytest.mark.parametrize("kind", KINDS)
def test_stale_quote_blocks_every_order_type(rm, store, kind):
    prep(store, kind)
    d = check(rm, make(kind), q=quote(NOON, age_minutes=26))    # limit is 15 + 10 = 25 minutes
    only_reason(d, "stale")


@pytest.mark.parametrize("kind", KINDS)
def test_quote_at_exactly_the_stale_limit_is_fresh(rm, store, kind):
    prep(store, kind)
    assert check(rm, make(kind), q=quote(NOON, age_minutes=25)).approved


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_quote_without_a_usable_price_blocks(rm, bad):
    d = check(rm, buy(), q=quote(NOON, price=bad))
    assert not d.approved
    assert any("usable price" in r for r in d.reasons), d.reasons


def test_quote_without_timezone_is_rejected_not_raised(rm):
    naive = Quote(symbol=SYMBOL, price=280.0, ts=datetime(2026, 11, 2, 11, 59))
    d = check(rm, buy(), q=naive)
    assert not d.approved
    assert any("timezone" in r for r in d.reasons), d.reasons


# ---------------------------------------------------------------- universe


def test_symbol_not_in_universe_blocks_entry(rm):
    only_reason(check(rm, buy(symbol="TCS"), i=idea(symbol="TCS")), "universe")


@pytest.mark.parametrize("kind", ["sell", "exit"])
def test_universe_does_not_apply_to_sells_and_exits(rm, store, kind):
    hold(store, symbol="TCS")
    assert check(rm, make(kind, symbol="TCS")).approved


# ---------------------------------------------------------------- quantity


@pytest.mark.parametrize("qty", [0, -2])
@pytest.mark.parametrize("kind", KINDS)
def test_qty_below_one_blocks_every_order_type(rm, store, kind, qty):
    prep(store, kind)
    d = check(rm, make(kind, qty=qty))
    assert not d.approved
    assert any("quantity" in r for r in d.reasons), d.reasons


def test_fractional_qty_blocks_entry(rm):
    d = check(rm, buy(qty=2.5))
    assert any("quantity" in r for r in d.reasons), d.reasons
    assert not d.approved


@pytest.mark.parametrize("kind", ["sell", "exit"])
def test_sell_qty_above_held_blocks(rm, store, kind):
    hold(store, qty=3)
    only_reason(check(rm, make(kind, qty=4)), "exceeds the 3 held")


@pytest.mark.parametrize("kind", ["sell", "exit"])
def test_selling_something_not_held_blocks(rm, kind):
    only_reason(check(rm, make(kind)), "exceeds the 0 held")


@pytest.mark.parametrize("kind", ["sell", "exit"])
def test_sell_of_exactly_the_held_qty_is_allowed(rm, store, kind):
    hold(store, qty=3)
    assert check(rm, make(kind, qty=3)).approved


# ---------------------------------------------------------------- limit price


@pytest.mark.parametrize("limit", [278.6, 280.0, 281.4])
def test_entry_limit_inside_band_is_allowed(rm, limit):
    assert check(rm, buy(limit=limit)).approved


@pytest.mark.parametrize("limit", [278.5, 281.45, 300.0, 250.0])
def test_limit_band_blocks_entry(rm, limit):
    only_reason(check(rm, buy(limit=limit)), "outside ±0.5%")


@pytest.mark.parametrize("limit", [278.6, 280.0, 281.4])
def test_discretionary_sell_limit_inside_band_is_allowed(rm, store, limit):
    hold(store)
    assert check(rm, sell(limit=limit)).approved


@pytest.mark.parametrize("limit", [278.5, 281.45])
def test_limit_band_blocks_discretionary_sell(rm, store, limit):
    hold(store)
    only_reason(check(rm, sell(limit=limit)), "outside ±0.5%")


@pytest.mark.parametrize("limit", [277.2, 279.0, 280.0, 285.0])
def test_exit_limit_at_or_above_one_percent_through_is_allowed(rm, store, limit):
    hold(store)
    assert check(rm, exit_(limit=limit)).approved


@pytest.mark.parametrize("limit", [277.19, 270.0, 1.0])
def test_exit_band_blocks_limit_more_than_one_percent_below(rm, store, limit):
    hold(store)
    only_reason(check(rm, exit_(limit=limit)), "below last price")


@pytest.mark.parametrize("bad", [0.0, -5.0, math.nan, math.inf])
@pytest.mark.parametrize("kind", KINDS)
def test_unusable_limit_price_blocks_without_raising(rm, store, kind, bad):
    prep(store, kind)
    d = check(rm, make(kind, limit=bad))
    assert not d.approved
    assert any("limit price" in r for r in d.reasons), d.reasons


# ---------------------------------------------------------------- limit state


def test_limit_state_blocks_entry_when_daily_paused(rm, limits):
    assert limits[BOOK].evaluate(850.0, NOON).state is LimitState.DAILY_PAUSED
    only_reason(check(rm, buy()), "daily loss limit")


def test_limit_state_blocks_entry_when_hard_stopped(rm, limits):
    assert limits[BOOK].evaluate(450.0, NOON).state is LimitState.HARD_STOPPED
    only_reason(check(rm, buy()), "hard stop")


def test_daily_pause_ends_next_trading_day(rm, limits):
    limits[BOOK].evaluate(850.0, NOON)
    assert check(rm, buy(), now=at(TUE, "12:00")).approved


def test_limit_state_is_per_book(rm, limits):
    limits[BOOK].evaluate(450.0, NOON)
    d = check(rm, buy(book=LEARNING, qty=3))
    assert d.approved, d.reasons


def test_sell_allowed_when_daily_paused(rm, store, limits):
    hold(store)
    limits[BOOK].evaluate(850.0, NOON)
    assert check(rm, sell()).approved


def test_exit_allowed_when_hard_stopped(rm, store, limits):
    hold(store)
    assert limits[BOOK].evaluate(450.0, NOON).state is LimitState.HARD_STOPPED
    d = check(rm, exit_())
    assert d.approved, d.reasons


# ---------------------------------------------------------------- entries per day


def test_max_entries_per_day_blocks_entry(rm, store):
    add_fill(store, Side.BUY, at(MON, "10:00"), symbol="GOLDBEES")
    only_reason(check(rm, buy()), "daily entry limit reached (1/1)")


def test_max_entries_per_day_counts_only_today(rm, store):
    add_fill(store, Side.BUY, at(FRI, "10:00"), symbol="GOLDBEES")
    assert check(rm, buy()).approved


def test_exit_fills_do_not_count_as_entries(rm, store):
    add_fill(store, Side.SELL, at(MON, "10:00"), is_exit=True, symbol="GOLDBEES")
    assert check(rm, buy()).approved


def test_learning_book_allows_three_entries_a_day(rm, store):
    for sym in ("GOLDBEES", "RELIANCE"):
        add_fill(store, Side.BUY, at(MON, "10:00"), symbol=sym, book=LEARNING)
    assert check(rm, buy(book=LEARNING)).approved
    add_fill(store, Side.BUY, at(MON, "11:00"), symbol="BANKBEES", book=LEARNING)
    only_reason(check(rm, buy(book=LEARNING)), "daily entry limit reached (3/3)")


@pytest.mark.parametrize("kind", ["sell", "exit"])
def test_entry_cap_does_not_apply_to_sells_and_exits(rm, store, kind):
    hold(store)
    add_fill(store, Side.BUY, at(MON, "10:00"), symbol="GOLDBEES")
    assert check(rm, make(kind)).approved


# ---------------------------------------------------------------- bought today


def test_bought_today_blocks_entry(rm, store):
    add_fill(store, Side.BUY, at(MON, "10:00"), book=LEARNING)
    only_reason(check(rm, buy(book=LEARNING)), f"already bought {SYMBOL} today")


def test_bought_today_ignores_other_symbols_and_other_days(rm, store):
    add_fill(store, Side.BUY, at(MON, "10:00"), symbol="GOLDBEES", book=LEARNING)
    add_fill(store, Side.BUY, at(FRI, "10:00"), book=LEARNING)
    assert check(rm, buy(book=LEARNING)).approved


# ---------------------------------------------------------------- position value


def test_position_cap_blocks_entry(rm):
    # learning: 20% of 100000 = 20000. 72 x 280 = 20160.
    d = check(rm, buy(book=LEARNING, qty=72))
    only_reason(d, "20% of equity")


def test_position_just_under_cap_is_allowed(rm):
    assert check(rm, buy(book=LEARNING, qty=71)).approved             # 71 x 280 = 19880


def test_position_exactly_at_cap_is_allowed(rm):
    q = quote(NOON, price=200.0)
    assert check(rm, buy(book=LEARNING, qty=100, limit=200.0), q=q).approved      # 20000.00
    only_reason(check(rm, buy(book=LEARNING, qty=101, limit=200.0), q=q), "20% of equity")


def test_position_cap_uses_equity_not_capital(rm, store):
    store.set_cash(LEARNING, 50000.0)                     # equity halved: cap is 10000
    only_reason(check(rm, buy(book=LEARNING, qty=40)), "20% of equity")        # 40 x 280 = 11200
    assert check(rm, buy(book=LEARNING, qty=35)).approved                      # 9800


def test_position_cap_includes_the_value_already_held(rm, store):
    # 60 x 200 held, cash 88000. Equity is a little under 100000 (sell charges), so the cap is
    # a little under 20000: adding 40 x 200 lands on 20000 and is refused.
    store.set_cash(LEARNING, 88000.0)
    hold(store, qty=60, avg_price=200.0, book=LEARNING)
    q = quote(NOON, price=200.0)
    assert check(rm, buy(book=LEARNING, qty=39, limit=200.0), q=q).approved
    only_reason(check(rm, buy(book=LEARNING, qty=40, limit=200.0), q=q), "20% of equity")


def test_realistic_book_cap_is_its_whole_equity(rm):
    assert check(rm, buy(qty=3)).approved                                      # 840 of ~1000
    only_reason(check(rm, buy(qty=4)), "100% of equity")                       # 1120


def test_position_cap_is_not_applied_to_sells_and_exits(rm, store):
    hold(store, qty=100, book=LEARNING)
    for o in (sell(book=LEARNING, qty=100), exit_(book=LEARNING, qty=100)):
        assert check(rm, o).approved


# ---------------------------------------------------------------- cost rule


def test_cost_rule_blocks_tiny_move(rm):
    # 3 x NIFTYBEES @ 280 with a 2% expected move: 16.8 < 3 x ~17.3
    d = check(rm, buy(qty=3, limit=280.0), i=idea(move_pct=2.0))
    only_reason(d, "round-trip charges")
    cost = round_trip_charges(3, 280.0, 280.0, CHARGES)
    assert 0.02 * 3 * 280.0 == pytest.approx(16.8)
    assert 17.0 < cost < 17.5


def test_cost_rule_passes_at_the_exact_boundary(rm):
    cost = round_trip_charges(3, 280.0, 280.0, CHARGES)
    needed_pct = SETTINGS.risk.cost_multiple * cost / (3 * 280.0) * 100
    assert check(rm, buy(), i=idea(move_pct=needed_pct)).approved
    assert check(rm, buy(), i=idea(move_pct=needed_pct * (1 - 1e-9))).approved     # float noise
    only_reason(check(rm, buy(), i=idea(move_pct=needed_pct - 0.001)), "round-trip charges")


def test_cost_rule_uses_the_limit_price_and_qty(rm):
    # same 8% move; a single unit at 280 cannot clear 3x the 15.34 DP charge
    only_reason(check(rm, buy(qty=1)), "round-trip charges")


def test_cost_rule_requires_an_idea_for_entries(rm):
    only_reason(check(rm, buy(), i=None), "trade idea")


@pytest.mark.parametrize("bad", [0.0, -3.0, math.nan])
def test_cost_rule_blocks_a_non_positive_expected_move(rm, bad):
    only_reason(check(rm, buy(), i=idea(move_pct=bad)), "round-trip charges")


@pytest.mark.parametrize("kind", ["sell", "exit"])
def test_cost_rule_does_not_apply_to_sells_and_exits(rm, store, kind):
    hold(store)
    assert check(rm, make(kind), i=None).approved
    assert check(rm, make(kind), i=idea(move_pct=0.01)).approved


# ---------------------------------------------------------------- sell-day cap


def three_sell_days(store, book=BOOK):
    for day in (MON, TUE, WED):
        add_fill(store, Side.SELL, at(day, "11:00"), book=book)


def test_sell_day_cap_blocks_discretionary_sell(rm, store):
    hold(store)
    three_sell_days(store)
    only_reason(check(rm, sell(), now=at(THU, "12:00")), "sell-day cap reached (3/3 days)")


def test_sell_day_cap_allows_the_third_day(rm, store):
    hold(store)
    for day in (MON, TUE):
        add_fill(store, Side.SELL, at(day, "11:00"))
    assert check(rm, sell(), now=at(WED, "12:00")).approved


def test_sell_day_cap_allows_more_sells_on_a_day_already_counted(rm, store):
    hold(store)
    three_sell_days(store)
    assert check(rm, sell(), now=at(WED, "14:00")).approved


def test_sell_day_cap_resets_on_monday(rm, store):
    hold(store)
    three_sell_days(store)
    assert check(rm, sell(), now=at(NEXT_MON, "12:00")).approved


def test_exit_fills_do_not_use_up_the_sell_day_cap(rm, store):
    hold(store)
    for day in (MON, TUE):
        add_fill(store, Side.SELL, at(day, "11:00"))
    add_fill(store, Side.SELL, at(WED, "11:00"), is_exit=True)
    assert check(rm, sell(), now=at(THU, "12:00")).approved


def test_exit_allowed_when_sell_cap_reached(rm, store):
    hold(store)
    three_sell_days(store)
    d = check(rm, exit_(), now=at(THU, "12:00"))
    assert d.approved, d.reasons


def test_sell_day_cap_does_not_apply_to_entries(rm, store):
    three_sell_days(store)
    assert check(rm, buy(), now=at(THU, "12:00")).approved


def test_learning_book_has_no_sell_day_cap(rm, store):
    hold(store, book=LEARNING)
    for day in (MON, TUE, WED, THU):
        add_fill(store, Side.SELL, at(day, "11:00"), book=LEARNING)
    assert check(rm, sell(book=LEARNING), now=at(date(2026, 11, 6), "12:00")).approved


def test_sell_day_cap_is_per_book(rm, store):
    hold(store, book=LEARNING)
    three_sell_days(store)                                  # realistic only
    assert check(rm, sell(book=LEARNING), now=at(THU, "12:00")).approved


# ---------------------------------------------------------------- classification and robustness


def test_buy_flagged_as_exit_is_still_treated_as_an_entry(rm):
    d = check(rm, buy(is_exit=True), now=at(MON, "09:15"))
    has_reason(d, "BUY cannot be an exit")
    has_reason(d, "outside entry window")


def test_buy_flagged_as_exit_cannot_dodge_the_entry_rules(rm, limits):
    limits[BOOK].evaluate(450.0, NOON)
    d = check(rm, buy(is_exit=True))
    has_reason(d, "hard stop")


def test_unknown_order_side_is_rejected(rm):
    o = order(side="SHORT")
    d = check(rm, o)
    assert not d.approved
    assert any("side" in r for r in d.reasons), d.reasons


@pytest.mark.parametrize("kind", KINDS)
def test_unknown_book_is_a_rejection_not_an_exception(rm, kind):
    d = check(rm, make(kind, book="ghost"))
    assert not d.approved
    assert any("unknown book 'ghost'" in r for r in d.reasons), d.reasons


def test_book_without_loss_limits_counts_as_unknown(store, limits):
    rm = RiskManager(store, SETTINGS, {BOOK: limits[BOOK]}, UNIVERSE)
    d = check(rm, buy(book=LEARNING))
    assert not d.approved
    assert any("unknown book 'learning'" in r for r in d.reasons), d.reasons


def test_unknown_book_still_reports_the_other_failures(rm):
    d = check(rm, buy(book="ghost", symbol="TCS"), now=at(MON, "09:15"), q=None)
    assert not d.approved
    assert len(d.reasons) >= 4, d.reasons


def test_book_with_no_cash_row_is_a_rejection_not_an_exception(tmp_path, limits):
    s = Store(tmp_path / "other.db")
    s.set_cash(BOOK, 1000.0)                               # learning never seeded
    try:
        rm = RiskManager(s, SETTINGS, limits, UNIVERSE)
        d = check(rm, buy(book=LEARNING))
        assert not d.approved
        assert any("cash" in r for r in d.reasons), d.reasons
    finally:
        s.close()


def test_collects_multiple_reasons(rm, store, limits):
    limits[BOOK].evaluate(450.0, NOON)                                     # hard stop
    add_fill(store, Side.BUY, at(MON, "09:16"), symbol="GOLDBEES")         # the day's one entry is used
    now = at(MON, "09:17")                                                 # before the entry window
    o = buy(symbol="TCS", qty=0, limit=250.0)                              # off-universe, no qty
    d = rm.check(o, quote(now, symbol="TCS", age_minutes=30), now, idea(move_pct=0.1))
    assert not d.approved
    joined = " | ".join(d.reasons)
    for fragment in ("entry window", "stale", "universe", "quantity", "outside ±0.5%", "hard stop",
                     "daily entry limit"):
        assert fragment in joined, (fragment, d.reasons)
    assert len(d.reasons) >= 7
    assert all(isinstance(r, str) and 0 < len(r) < 120 for r in d.reasons), d.reasons


def test_universe_is_copied_at_construction(store, limits):
    universe = {SYMBOL}
    rm = RiskManager(store, SETTINGS, limits, universe)
    universe.clear()
    assert check(rm, buy()).approved


def test_check_does_not_write_to_the_store(rm, store):
    before = (store.cash(BOOK), store.positions(BOOK), store.pending_orders(), store.risk_approved("o1"))
    check(rm, buy())
    check(rm, buy(qty=0))
    after = (store.cash(BOOK), store.positions(BOOK), store.pending_orders(), store.risk_approved("o1"))
    assert before == after


# ---------------------------------------------------------------- pending (approved, unfilled) orders


def test_pending_buy_counts_toward_the_entry_cap(rm, store):
    add_pending(store, Side.BUY, symbol="GOLDBEES")
    only_reason(check(rm, buy()), "daily entry limit reached (1/1)")


def test_pending_buys_and_fills_add_up_toward_the_entry_cap(rm, store):
    add_pending(store, Side.BUY, symbol="GOLDBEES", book=LEARNING)
    add_pending(store, Side.BUY, symbol="RELIANCE", book=LEARNING)
    assert check(rm, buy(book=LEARNING)).approved                       # 2 of 3
    add_fill(store, Side.BUY, at(MON, "10:00"), symbol="BANKBEES", book=LEARNING)
    only_reason(check(rm, buy(book=LEARNING)), "daily entry limit reached (3/3)")


def test_pending_buy_from_yesterday_does_not_count_toward_todays_entry_cap(rm, store):
    add_pending(store, Side.BUY, symbol="GOLDBEES", created=at(FRI, "11:00"))
    assert check(rm, buy()).approved


def test_pending_sells_other_books_and_closed_orders_do_not_count_as_entries(rm, store):
    add_pending(store, Side.SELL, symbol="GOLDBEES")
    add_pending(store, Side.BUY, symbol="GOLDBEES", book=LEARNING)
    add_pending(store, Side.BUY, symbol="GOLDBEES", status=OrderStatus.CANCELLED)
    add_pending(store, Side.BUY, symbol="GOLDBEES", status=OrderStatus.REJECTED)
    assert check(rm, buy()).approved


def test_pending_buy_of_the_same_symbol_blocks_a_rebuy(rm, store):
    add_pending(store, Side.BUY, book=LEARNING)
    only_reason(check(rm, buy(book=LEARNING)), f"already bought {SYMBOL} today")


def test_pending_buy_of_another_symbol_or_book_or_day_does_not_block_a_buy(rm, store):
    add_pending(store, Side.BUY, symbol="GOLDBEES", book=LEARNING)                       # other symbol
    add_pending(store, Side.BUY, book=BOOK)                                              # other book
    add_pending(store, Side.BUY, book=LEARNING, created=at(FRI, "11:00"))                # yesterday
    assert check(rm, buy(book=LEARNING)).approved


def test_position_cap_counts_pending_buy_value(rm, store):
    q = quote(NOON, price=200.0)
    # learning cap: 20% of 100000 = 20000. The pending buy is from yesterday, so only the cap sees it.
    assert check(rm, buy(book=LEARNING, qty=41, limit=200.0), q=q).approved            # 8200 alone fits
    add_pending(store, Side.BUY, qty=60, limit=200.0, book=LEARNING, created=at(FRI, "11:00"))  # 12000
    only_reason(check(rm, buy(book=LEARNING, qty=41, limit=200.0), q=q), "20% of equity")   # 20200
    assert check(rm, buy(book=LEARNING, qty=39, limit=200.0), q=q).approved                 # 19800


def test_position_cap_ignores_pending_buys_of_other_symbols_and_sells(rm, store):
    q = quote(NOON, price=200.0)
    add_pending(store, Side.BUY, qty=60, limit=200.0, symbol="GOLDBEES", book=LEARNING)
    add_pending(store, Side.SELL, qty=60, limit=200.0, book=LEARNING)
    assert check(rm, buy(book=LEARNING, qty=41, limit=200.0), q=q).approved


def test_two_sells_cannot_together_exceed_the_holding(rm, store):
    hold(store, qty=3)
    add_pending(store, Side.SELL, qty=3)
    only_reason(check(rm, sell(qty=3)), "3 already pending sale")
    only_reason(check(rm, exit_(qty=3)), "3 already pending sale")


def test_pending_sell_leaves_the_rest_of_the_holding_available(rm, store):
    hold(store, qty=3)
    add_pending(store, Side.SELL, qty=2)
    assert check(rm, sell(qty=1)).approved
    only_reason(check(rm, sell(qty=2)), "already pending sale")


def test_pending_exit_sell_blocks_a_discretionary_sell(rm, store):
    hold(store, qty=3)
    add_pending(store, Side.SELL, qty=3, is_exit=True)
    only_reason(check(rm, sell(qty=3)), "already pending sale")


def test_pending_sell_from_an_earlier_day_still_reserves_the_holding(rm, store):
    hold(store, qty=3)
    add_pending(store, Side.SELL, qty=3, created=at(FRI, "11:00"))
    only_reason(check(rm, sell(qty=3)), "already pending sale")


def test_pending_buys_other_symbols_and_other_books_do_not_reduce_what_can_be_sold(rm, store):
    hold(store, qty=3)
    add_pending(store, Side.BUY, qty=5)                                    # not owned yet
    add_pending(store, Side.SELL, qty=3, symbol="GOLDBEES")
    add_pending(store, Side.SELL, qty=3, book=LEARNING)
    assert check(rm, sell(qty=3)).approved
    assert check(rm, exit_(qty=3)).approved


def test_pending_sell_today_makes_today_a_sell_day(rm, store):
    hold(store, qty=6)
    three_sell_days(store)
    now = at(THU, "12:00")
    only_reason(check(rm, sell(), now=now), "sell-day cap")                     # control: no pending
    add_pending(store, Side.SELL, qty=3, created=at(THU, "10:00"))
    assert check(rm, sell(), now=now).approved                                  # 3 of 6 still free


def test_pending_exit_sell_today_does_not_make_today_a_sell_day(rm, store):
    hold(store, qty=6)
    three_sell_days(store)
    add_pending(store, Side.SELL, qty=3, is_exit=True, created=at(THU, "10:00"))
    only_reason(check(rm, sell(), now=at(THU, "12:00")), "sell-day cap")


def test_pending_sell_from_an_earlier_day_does_not_make_today_a_sell_day(rm, store):
    hold(store, qty=6)
    three_sell_days(store)
    add_pending(store, Side.SELL, qty=3, created=at(WED, "13:00"))
    only_reason(check(rm, sell(), now=at(THU, "12:00")), "sell-day cap")


def test_order_under_check_is_not_counted_against_itself_for_the_entry_cap(rm, store):
    o = buy(id="retry")
    store.save_order(o)                                  # approved earlier, still PENDING
    d = check(rm, o)
    assert d.approved, d.reasons


def test_order_under_check_is_not_counted_against_itself_for_bought_today(rm, store):
    o = buy(book=LEARNING, id="retry")
    store.save_order(o)
    assert check(rm, o).approved


def test_order_under_check_is_not_counted_against_itself_for_the_position_cap(rm, store):
    o = buy(book=LEARNING, qty=71, id="retry")           # 19880 fits under 20000; counted twice it would not
    store.save_order(o)
    assert check(rm, o).approved


@pytest.mark.parametrize("kind", ["sell", "exit"])
def test_order_under_check_is_not_counted_against_itself_for_the_held_qty(rm, store, kind):
    hold(store, qty=3)
    o = make(kind, qty=3, id="retry")
    store.save_order(o)
    assert check(rm, o).approved


def test_order_under_check_does_not_make_today_a_sell_day_for_itself(rm, store):
    hold(store, qty=3)
    three_sell_days(store)
    now = at(THU, "12:00")
    o = sell(qty=3, id="retry", created_at=now)
    store.save_order(o)
    only_reason(check(rm, o, now=now), "sell-day cap")


def test_idea_for_a_different_symbol_is_rejected(rm):
    only_reason(check(rm, buy(), i=idea(symbol="GOLDBEES")), "idea is for a different symbol")
