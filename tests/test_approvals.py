import logging
import threading
import time
from datetime import date, datetime, timedelta

import pytest

from aitrader.approvals import (ApprovalBook, ApprovalCard, Resolution, get_autopilot,
                                set_autopilot)
from aitrader.calendar import IST
from aitrader.charges import compute_charges
from aitrader.config import ApprovalConfig, ChargesConfig
from aitrader.models import Side, TradeIdea
from aitrader.store import Store

# The values in config.yaml.
APPROVAL = ApprovalConfig(expiry_minutes=10, reprice_tolerance_pct=0.5)
CHARGES = ChargesConfig(brokerage_per_order=0.0, stt_pct=0.1, stamp_buy_pct=0.015, exchange_pct=0.00322,
                        sebi_pct=0.0001, gst_pct=18.0, dp_per_sell=15.34)
DAY = date(2026, 11, 2)  # a Monday


def t(hhmm: str, seconds: int = 0) -> datetime:
    h, m = hhmm.split(":")
    return datetime(DAY.year, DAY.month, DAY.day, int(h), int(m), seconds, tzinfo=IST)


def make_idea(side=Side.BUY, limit_price=280.0, id="i1"):
    return TradeIdea(id=id, symbol="NIFTYBEES", side=side, limit_price=limit_price, stop=270.0,
                     target=300.0, expected_move_pct=3.5, confidence=0.7, holding_days=5,
                     reason="trend is up, volume rising")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


@pytest.fixture
def book(store):
    return ApprovalBook(store, APPROVAL, "15:10")


SIZES = {"realistic": 3, "learning": 71}


def new_card(book, quote=280.0, now=None, sizes=None):
    return book.create(make_idea(), SIZES if sizes is None else sizes, quote, now or t("10:00"))


# ---- autopilot ----

def test_autopilot_defaults_off(store):
    assert get_autopilot(store) is False


def test_autopilot_set_and_clear(store):
    set_autopilot(store, True)
    assert get_autopilot(store) is True
    set_autopilot(store, False)
    assert get_autopilot(store) is False


def test_autopilot_survives_reopen(tmp_path):
    s = Store(tmp_path / "t.db")
    set_autopilot(s, True)
    s.close()
    s2 = Store(tmp_path / "t.db")
    assert get_autopilot(s2) is True
    s2.close()


@pytest.mark.parametrize("stored", [1, "true", "True", "yes", "on", None, 0, [True], {"on": True}])
def test_autopilot_only_literal_true_counts_as_on(store, stored):
    store.set_state("autopilot", stored)
    assert get_autopilot(store) is False


def test_set_autopilot_rejects_non_bool(store):
    with pytest.raises(TypeError):
        set_autopilot(store, "on")  # type: ignore[arg-type]
    assert get_autopilot(store) is False


# ---- create ----

def test_expiry_is_min_of_10min_and_1510(book):
    assert new_card(book, now=t("15:05")).expires_at == t("15:10")
    assert new_card(book, now=t("10:00")).expires_at == t("10:10")
    assert new_card(book, now=t("15:00")).expires_at == t("15:10")   # 10 minutes exactly hits the cap
    assert new_card(book, now=t("15:00", 1)).expires_at == t("15:10")


def test_expiry_when_created_after_entry_end_is_the_entry_end(book):
    card = new_card(book, now=t("15:20"))
    assert card.expires_at == t("15:10")
    assert book.resolve(card.id, True, 280.0, t("15:20")) is Resolution.EXPIRED


def test_expiry_uses_the_ist_date_of_now(book):
    # 23:00 UTC on 2 Nov is 04:30 IST on 3 Nov: the cap is 3 Nov 15:10 IST, so 10 minutes wins.
    from datetime import timezone
    now_utc = datetime(2026, 11, 2, 23, 0, tzinfo=timezone.utc)
    card = new_card(book, now=now_utc)
    assert card.expires_at == datetime(2026, 11, 3, 4, 40, tzinfo=IST)
    assert card.expires_at.utcoffset() == timedelta(hours=5, minutes=30)


def test_create_returns_an_open_card_with_the_given_fields(book):
    card = new_card(book, quote=281.5)
    assert isinstance(card, ApprovalCard)
    assert card.status == "open"
    assert card.idea_id == "i1"
    assert card.sizes == SIZES
    assert card.quote_at_issue == 281.5
    assert len(card.id) == 32 and int(card.id, 16) >= 0   # uuid4 hex


def test_card_ids_are_unique(book):
    assert len({new_card(book).id for _ in range(50)}) == 50


def test_create_copies_sizes(book):
    sizes = {"realistic": 3, "learning": 0}
    card = new_card(book, sizes=sizes)
    sizes["learning"] = 99
    assert card.sizes == {"realistic": 3, "learning": 0}
    assert book.get(card.id).sizes == {"realistic": 3, "learning": 0}


def test_payload_is_json_with_iso_offset_datetimes_and_int_sizes(book, store):
    card = new_card(book, now=t("10:00"))
    payload = store.get_card(card.id)
    assert payload["status"] == "open"
    assert payload["idea_id"] == "i1"
    assert payload["sizes"] == {"realistic": 3, "learning": 71}
    assert payload["quote_at_issue"] == 280.0
    assert isinstance(payload["expires_at"], str)
    assert payload["expires_at"].endswith("+05:30")
    assert datetime.fromisoformat(payload["expires_at"]) == t("10:10")


def test_get_rebuilds_the_card_with_aware_ist_datetime(book):
    card = new_card(book)
    got = book.get(card.id)
    assert got == card
    assert got.expires_at.tzinfo is not None
    assert got.expires_at.utcoffset() == timedelta(hours=5, minutes=30)
    assert book.get("nope") is None


def test_cards_survive_a_restart(tmp_path):
    s = Store(tmp_path / "t.db")
    card = ApprovalBook(s, APPROVAL, "15:10").create(make_idea(), SIZES, 280.0, t("10:00"))
    s.close()
    s2 = Store(tmp_path / "t.db")
    book2 = ApprovalBook(s2, APPROVAL, "15:10")
    assert book2.get(card.id) == card
    assert book2.resolve(card.id, True, 280.0, t("10:05")) is Resolution.APPROVED
    s2.close()


def test_create_saves_the_idea_so_the_card_can_be_rendered_after_a_restart(book, store):
    idea = make_idea()
    book.create(idea, SIZES, 280.0, t("10:00"))
    assert store.get_idea("i1") == idea


@pytest.mark.parametrize("sizes", [
    {},                                   # no books
    {"realistic": 0, "learning": 0},      # nothing to buy
    {"realistic": -1, "learning": 5},     # negative
    {"realistic": 1.5},                   # not an int
    {"realistic": 2.0},                   # a float, even a whole one
    {"realistic": True},                  # bool is not a quantity
    {"realistic": "3"},
    {"realistic": None},
    {1: 3},                               # book names are strings
])
def test_create_rejects_bad_sizes(book, store, sizes):
    with pytest.raises(ValueError):
        book.create(make_idea(), sizes, 280.0, t("10:00"))
    assert store.cards_with_status("open") == []


def test_create_accepts_a_zero_size_for_one_book(book):
    card = new_card(book, sizes={"realistic": 0, "learning": 12})
    assert card.sizes == {"realistic": 0, "learning": 12}


def test_create_rejects_naive_now(book):
    with pytest.raises(ValueError):
        book.create(make_idea(), SIZES, 280.0, datetime(2026, 11, 2, 10, 0))


@pytest.mark.parametrize("quote", [0.0, -1.0, float("nan"), float("inf"), None])
def test_create_rejects_unusable_quote(book, quote):
    with pytest.raises(ValueError):
        book.create(make_idea(), SIZES, quote, t("10:00"))


# ---- resolve ----

def test_approve_within_tolerance(book):
    card = new_card(book, quote=280.0)
    assert book.resolve(card.id, True, 280.5, t("10:03")) is Resolution.APPROVED   # 0.18%
    assert book.get(card.id).status == "approved"


def test_approve_at_exactly_the_tolerance_does_not_reissue(book):
    a = new_card(book, quote=100.0)
    assert book.resolve(a.id, True, 100.5, t("10:03")) is Resolution.APPROVED
    # 201 / 200 - 1 is 0.005000000000000115 in floating point; it is still exactly 0.5%.
    b = new_card(book, quote=200.0)
    assert book.resolve(b.id, True, 201.0, t("10:03")) is Resolution.APPROVED
    c = new_card(book, quote=100.0)
    assert book.resolve(c.id, True, 99.5, t("10:03")) is Resolution.APPROVED


def test_reissue_when_price_moved(book):
    card = new_card(book, quote=100.0)
    assert book.resolve(card.id, True, 100.6, t("10:03")) is Resolution.REISSUE   # 0.6%
    assert book.get(card.id).status == "expired"


def test_reissue_when_price_fell(book):
    card = new_card(book, quote=100.0)
    assert book.resolve(card.id, True, 99.4, t("10:03")) is Resolution.REISSUE
    assert book.get(card.id).status == "expired"


def test_tap_on_a_reissued_card_again_reports_expired(book):
    card = new_card(book, quote=100.0)
    assert book.resolve(card.id, True, 103.0, t("10:03")) is Resolution.REISSUE
    assert book.resolve(card.id, True, 100.0, t("10:04")) is Resolution.EXPIRED
    assert book.resolve(card.id, False, 100.0, t("10:04")) is Resolution.EXPIRED


@pytest.mark.parametrize("price", [0.0, -5.0, float("nan"), float("inf"), float("-inf"), None, "abc"])
def test_reissue_when_current_price_cannot_be_confirmed(book, caplog, price):
    card = new_card(book, quote=100.0)
    with caplog.at_level(logging.WARNING, logger="aitrader.approvals"):
        assert book.resolve(card.id, True, price, t("10:03")) is Resolution.REISSUE
    assert book.get(card.id).status == "expired"
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_reject_open_card(book):
    card = new_card(book)
    assert book.resolve(card.id, False, 280.0, t("10:03")) is Resolution.REJECTED
    assert book.get(card.id).status == "rejected"


def test_reject_needs_no_price_check(book):
    card = new_card(book, quote=100.0)
    assert book.resolve(card.id, False, float("nan"), t("10:03")) is Resolution.REJECTED
    other = new_card(book, quote=100.0)
    assert book.resolve(other.id, False, 150.0, t("10:03")) is Resolution.REJECTED   # +50%


def test_double_approve_is_idempotent(book):
    card = new_card(book)
    assert book.resolve(card.id, True, 280.0, t("10:03")) is Resolution.APPROVED
    assert book.resolve(card.id, True, 280.0, t("10:03", 1)) is Resolution.ALREADY_RESOLVED
    assert book.get(card.id).status == "approved"


def test_a_resolved_card_stays_as_it_was_whatever_is_tapped_next(book):
    approved = new_card(book)
    assert book.resolve(approved.id, True, 280.0, t("10:03")) is Resolution.APPROVED
    assert book.resolve(approved.id, False, 280.0, t("10:04")) is Resolution.ALREADY_RESOLVED
    assert book.get(approved.id).status == "approved"

    rejected = new_card(book)
    assert book.resolve(rejected.id, False, 280.0, t("10:03")) is Resolution.REJECTED
    assert book.resolve(rejected.id, True, 280.0, t("10:04")) is Resolution.ALREADY_RESOLVED
    assert book.resolve(rejected.id, False, 280.0, t("10:04")) is Resolution.ALREADY_RESOLVED
    assert book.get(rejected.id).status == "rejected"


def test_a_resolved_card_is_already_resolved_even_after_its_expiry_time(book):
    card = new_card(book, now=t("10:00"))
    assert book.resolve(card.id, True, 280.0, t("10:03")) is Resolution.APPROVED
    assert book.resolve(card.id, True, 280.0, t("11:00")) is Resolution.ALREADY_RESOLVED


def test_approve_after_expiry(book):
    card = new_card(book, now=t("10:00"))
    assert book.resolve(card.id, True, 280.0, t("10:11")) is Resolution.EXPIRED
    assert book.get(card.id).status == "expired"


def test_expiry_wins_over_a_price_move(book):
    card = new_card(book, quote=100.0, now=t("10:00"))
    assert book.resolve(card.id, True, 150.0, t("10:11")) is Resolution.EXPIRED


def test_card_is_expired_from_its_expiry_time_on(book):
    a = new_card(book, now=t("10:00"))                       # expires 10:10:00
    assert book.resolve(a.id, True, 280.0, t("10:09", 59)) is Resolution.APPROVED
    b = new_card(book, now=t("10:00"))
    assert book.resolve(b.id, True, 280.0, t("10:10")) is Resolution.EXPIRED


def test_reject_after_expiry_reports_expired(book):
    # An open card past its time is closed, not rejected, even if the sweep has not run yet.
    card = new_card(book, now=t("10:00"))
    assert book.resolve(card.id, False, 280.0, t("10:11")) is Resolution.EXPIRED
    assert book.get(card.id).status == "expired"


def test_tapping_an_already_expired_card_reports_expired(book):
    card = new_card(book, now=t("10:00"))
    book.expire_due(t("10:11"))
    assert book.resolve(card.id, True, 280.0, t("10:12")) is Resolution.EXPIRED
    assert book.resolve(card.id, False, 280.0, t("10:12")) is Resolution.EXPIRED


def test_unknown_card(book):
    assert book.resolve("does-not-exist", True, 280.0, t("10:03")) is Resolution.UNKNOWN
    assert book.resolve("does-not-exist", False, 280.0, t("10:03")) is Resolution.UNKNOWN


def test_resolve_rejects_naive_now_and_leaves_the_card_open(book):
    card = new_card(book)
    with pytest.raises(ValueError):
        book.resolve(card.id, True, 280.0, datetime(2026, 11, 2, 10, 3))
    assert book.get(card.id).status == "open"


def test_concurrent_taps_resolve_exactly_once(book, store, monkeypatch):
    card = new_card(book)
    real_get = store.get_card

    def slow_get(card_id):
        payload = real_get(card_id)
        time.sleep(0.02)   # widen the window between the read and the write
        return payload

    monkeypatch.setattr(store, "get_card", slow_get)
    n = 12
    barrier = threading.Barrier(n)
    results: list[Resolution] = []
    errors: list[BaseException] = []

    def tap(approve: bool):
        try:
            barrier.wait()
            results.append(book.resolve(card.id, approve, 280.0, t("10:03")))
        except BaseException as e:   # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=tap, args=(i % 2 == 0,)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == []
    winners = [r for r in results if r in (Resolution.APPROVED, Resolution.REJECTED)]
    assert len(winners) == 1
    assert results.count(Resolution.ALREADY_RESOLVED) == n - 1
    assert book.get(card.id).status == ("approved" if winners[0] is Resolution.APPROVED else "rejected")


def test_a_failed_write_leaves_the_card_open(book, store, monkeypatch):
    # The read-check-write runs in one transaction: a failed write changes nothing.
    card = new_card(book)
    real = store.save_card

    def boom(card_id, payload):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "save_card", boom)
    with pytest.raises(RuntimeError):
        book.resolve(card.id, True, 280.0, t("10:03"))
    monkeypatch.setattr(store, "save_card", real)
    assert book.get(card.id).status == "open"
    assert book.resolve(card.id, True, 280.0, t("10:03")) is Resolution.APPROVED


# ---- expire_due ----

def test_expire_due_marks_overdue_open_cards(book):
    old = new_card(book, now=t("10:00"))          # expires 10:10
    fresh = new_card(book, now=t("10:08"))        # expires 10:18
    approved = new_card(book, now=t("10:00"))
    assert book.resolve(approved.id, True, 280.0, t("10:02")) is Resolution.APPROVED
    rejected = new_card(book, now=t("10:00"))
    assert book.resolve(rejected.id, False, 280.0, t("10:02")) is Resolution.REJECTED

    expired = book.expire_due(t("10:12"))

    assert [c.id for c in expired] == [old.id]
    assert all(c.status == "expired" for c in expired)
    assert book.get(old.id).status == "expired"
    assert book.get(fresh.id).status == "open"
    assert book.get(approved.id).status == "approved"
    assert book.get(rejected.id).status == "rejected"


def test_expire_due_is_idempotent(book):
    new_card(book, now=t("10:00"))
    assert len(book.expire_due(t("10:11"))) == 1
    assert book.expire_due(t("10:11")) == []


def test_expire_due_with_nothing_open(book):
    assert book.expire_due(t("10:11")) == []


def test_expire_due_expires_every_overdue_card(book):
    ids = {new_card(book, now=t("10:00")).id for _ in range(5)}
    assert {c.id for c in book.expire_due(t("15:30"))} == ids


def test_expire_due_rejects_naive_now(book):
    with pytest.raises(ValueError):
        book.expire_due(datetime(2026, 11, 2, 10, 11))


# ---- render ----

def test_render_contains_reason_and_sizes(book):
    idea = make_idea()
    card = book.create(idea, SIZES, 280.0, t("15:05"))
    text = book.render(card, idea, CHARGES)
    assert "NIFTYBEES" in text
    assert "BUY" in text
    assert "trend is up, volume rising" in text
    assert "realistic" in text and "learning" in text
    assert "3" in text and "71" in text
    assert "280.00" in text
    assert "300.00" in text      # target
    assert "270.00" in text      # stop
    assert "15:10 IST" in text   # expiry, wall clock


def test_render_shows_each_books_quantity_and_buy_charges(book):
    idea = make_idea()
    card = book.create(idea, SIZES, 280.0, t("10:00"))
    lines = book.render(card, idea, CHARGES).splitlines()
    for name, qty in SIZES.items():
        charges = compute_charges(Side.BUY, qty, 280.0, CHARGES).total
        line = next(ln for ln in lines if ln.startswith(name))
        assert f"{qty}" in line
        assert f"{charges:.2f}" in line


def test_render_skips_books_with_size_zero(book):
    idea = make_idea()
    card = book.create(idea, {"realistic": 0, "learning": 12}, 280.0, t("10:00"))
    text = book.render(card, idea, CHARGES)
    assert "realistic" not in text
    assert "learning" in text and "12" in text


def test_render_sell_uses_sell_charges_including_the_dp_charge(book):
    idea = make_idea(side=Side.SELL)
    card = book.create(idea, {"learning": 10}, 280.0, t("10:00"))
    text = book.render(card, idea, CHARGES)
    sell = compute_charges(Side.SELL, 10, 280.0, CHARGES).total
    buy = compute_charges(Side.BUY, 10, 280.0, CHARGES).total
    assert "SELL" in text
    assert f"{sell:.2f}" in text
    assert sell != pytest.approx(buy, abs=0.005)   # the DP charge makes them differ
    assert f"{buy:.2f}" not in text


def test_render_is_plain_text_with_two_decimal_rupees(book):
    idea = make_idea(limit_price=1234.5)
    idea.target = 1300.0
    idea.stop = 1200.0
    card = book.create(idea, {"learning": 5}, 1234.5, t("10:00"))
    text = book.render(card, idea, CHARGES)
    assert "₹1234.50" in text
    assert "₹1300.00" in text
    assert "₹1200.00" in text
    for markup in ("*", "<b>", "</", "`", "&lt;"):
        assert markup not in text


def test_render_shows_expiry_in_ist_whatever_the_card_timezone(book):
    idea = make_idea()
    card = book.create(idea, SIZES, 280.0, t("10:00"))
    assert "10:10 IST" in book.render(card, idea, CHARGES)


def test_render_survives_an_unusable_limit_price(book):
    idea = make_idea(limit_price=0.0)
    card = book.create(idea, SIZES, 280.0, t("10:00"))
    text = book.render(card, idea, CHARGES)
    assert "NIFTYBEES" in text and "trend is up" in text
