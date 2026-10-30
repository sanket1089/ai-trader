"""Approval cards (the owner taps approve or reject on Telegram) and the autopilot switch.

A card is a JSON dict in the store, so it survives a restart: datetimes are ISO strings with the
``+05:30`` offset and ``sizes`` is ``{book name: units}``. ``ApprovalCard`` is rebuilt from the
payload on every read, so a card object is a snapshot, never the source of truth.

Life of a card::

    open --approve--> approved          (price still within tolerance)
         --reject---> rejected
         --approve--> expired + REISSUE (price moved, or can't be confirmed: caller issues a new card)
         --time-----> expired           (resolve or expire_due, from ``expires_at`` on)
         --/stop----> expired           (expire_all_open: the owner's kill switch)

``resolve`` does its read, check and write inside one store transaction, so a double tap, or two
taps at the same moment, resolve the card exactly once; the loser gets ``ALREADY_RESOLVED``.
"""

import logging
import math
import numbers
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Literal

from aitrader.calendar import IST, at
from aitrader.charges import compute_charges
from aitrader.config import ApprovalConfig, ChargesConfig
from aitrader.models import Side, TradeIdea
from aitrader.store import Store

log = logging.getLogger(__name__)

AUTOPILOT_KEY = "autopilot"
CardStatus = Literal["open", "approved", "rejected", "expired"]

# Float slack so a move of exactly the tolerance (0.5%) never trips the re-issue: 201 / 200 - 1 is
# 0.005000000000000115 in floating point.
_MOVE_EPSILON = 1e-9


class Resolution(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    ALREADY_RESOLVED = "ALREADY_RESOLVED"
    REISSUE = "REISSUE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ApprovalCard:
    id: str
    idea_id: str
    sizes: dict[str, int]
    quote_at_issue: float
    expires_at: datetime
    status: CardStatus


# ---- autopilot ----

def get_autopilot(store: Store) -> bool:
    """True only if the stored value is exactly ``True``. Anything else, including a missing or
    odd value, is off: the safe answer is "ask the owner first"."""
    return store.get_state(AUTOPILOT_KEY, False) is True


def set_autopilot(store: Store, on: bool) -> None:
    if not isinstance(on, bool):
        raise TypeError(f"autopilot must be True or False, got {on!r}")
    store.set_state(AUTOPILOT_KEY, on)


# ---- helpers ----

def _require_aware(dt: datetime, name: str) -> None:
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime, got {dt!r}")


def _usable_price(x: object) -> bool:
    """A real number (not a bool) that is finite and above 0."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return False
    return math.isfinite(x) and x > 0


def _checked_sizes(sizes: Mapping[str, int]) -> dict[str, int]:
    """A copy of ``sizes``: book names are strings, units are ints >= 0, at least one above 0."""
    if not isinstance(sizes, Mapping):
        raise ValueError(f"sizes must be a mapping of book name to units, got {sizes!r}")
    clean: dict[str, int] = {}
    for book, qty in sizes.items():
        if not isinstance(book, str) or not book:
            raise ValueError(f"book names must be non-empty strings, got {book!r}")
        if isinstance(qty, bool) or not isinstance(qty, int) or qty < 0:
            raise ValueError(f"size for {book!r} must be a whole number >= 0, got {qty!r}")
        clean[book] = qty
    if not any(qty > 0 for qty in clean.values()):
        raise ValueError("at least one book needs a size above 0")
    return clean


def _to_payload(card: ApprovalCard) -> dict:
    return {
        "id": card.id,
        "idea_id": card.idea_id,
        "sizes": dict(card.sizes),
        "quote_at_issue": card.quote_at_issue,
        "expires_at": card.expires_at.astimezone(IST).isoformat(),
        "status": card.status,
    }


def _from_payload(p: dict) -> ApprovalCard:
    return ApprovalCard(
        id=p["id"],
        idea_id=p["idea_id"],
        sizes={str(book): int(qty) for book, qty in p["sizes"].items()},
        quote_at_issue=float(p["quote_at_issue"]),
        expires_at=datetime.fromisoformat(p["expires_at"]).astimezone(IST),
        status=p["status"],
    )


def _rupees(x: float) -> str:
    return f"₹{x:.2f}"


class ApprovalBook:
    def __init__(self, store: Store, approval_cfg: ApprovalConfig, entry_end: str):
        self._store = store
        self._cfg = approval_cfg
        self._entry_end = entry_end
        at(date(2000, 1, 3), entry_end)   # fail now, not at the first card, if entry_end is not "HH:MM"

    def create(self, idea: TradeIdea, sizes: Mapping[str, int], quote: float, now: datetime) -> ApprovalCard:
        """Open a new card. It expires ``expiry_minutes`` after ``now`` or at ``entry_end`` on
        today's IST date, whichever is first (a card made after ``entry_end`` is born expired).

        The idea is saved with the card so the card can always be rendered or re-issued later.
        """
        _require_aware(now, "now")
        clean = _checked_sizes(sizes)
        if not _usable_price(quote):
            raise ValueError(f"quote must be a finite number above 0, got {quote!r}")
        now = now.astimezone(IST)
        card = ApprovalCard(
            id=uuid.uuid4().hex,
            idea_id=idea.id,
            sizes=clean,
            quote_at_issue=float(quote),
            expires_at=min(now + timedelta(minutes=self._cfg.expiry_minutes),
                           at(now.date(), self._entry_end)),
            status="open",
        )
        with self._store.transaction():
            self._store.save_idea(idea)
            self._store.save_card(card.id, _to_payload(card))
        return card

    def get(self, card_id: str) -> ApprovalCard | None:
        payload = self._store.get_card(card_id)
        return _from_payload(payload) if payload is not None else None

    def resolve(self, card_id: str, approve: bool, current_price: float, now: datetime) -> Resolution:
        """Apply an owner tap exactly once.

        * unknown id: ``UNKNOWN``.
        * an expired card, for either tap: ``EXPIRED``. Any other closed card: ``ALREADY_RESOLVED``.
        * an open card at or past ``expires_at``: closed as expired, ``EXPIRED``, for either tap.
        * reject: ``REJECTED``, no price check.
        * approve: if ``current_price`` is not a finite number above 0, or is more than the
          tolerance away from the price at issue, the card is closed as expired and ``REISSUE`` is
          returned (the caller sends a fresh card). Otherwise ``APPROVED``.
        """
        _require_aware(now, "now")
        with self._store.transaction():   # read, check and write as one step: see the module docstring
            card = self.get(card_id)
            if card is None:
                return Resolution.UNKNOWN
            if card.status == "expired":
                return Resolution.EXPIRED
            if card.status != "open":
                return Resolution.ALREADY_RESOLVED
            if now >= card.expires_at:
                self._set_status(card, "expired")
                return Resolution.EXPIRED
            if not approve:
                self._set_status(card, "rejected")
                return Resolution.REJECTED
            if not self._price_confirmed(card, current_price):
                self._set_status(card, "expired")
                return Resolution.REISSUE
            self._set_status(card, "approved")
            return Resolution.APPROVED

    def expire_due(self, now: datetime) -> list[ApprovalCard]:
        """Close every open card whose ``expires_at`` has been reached. Returns those cards
        (status ``expired``), oldest first, so the caller can tell the owner."""
        _require_aware(now, "now")
        expired: list[ApprovalCard] = []
        with self._store.transaction():
            for payload in self._store.cards_with_status("open"):
                card = _from_payload(payload)
                if now >= card.expires_at:
                    expired.append(self._set_status(card, "expired"))
        return expired

    def expire_all_open(self, now: datetime) -> list[ApprovalCard]:
        """Close every open card as expired, whatever its time (the owner's /stop). A later tap on
        one gets ``EXPIRED`` and places nothing. Returns those cards, oldest first."""
        _require_aware(now, "now")
        with self._store.transaction():
            return [self._set_status(_from_payload(p), "expired") for p in self._store.cards_with_status("open")]

    def render(self, card: ApprovalCard, idea: TradeIdea, charges_cfg: ChargesConfig) -> str:
        """Plain text for the Telegram card (the adapter sends no parse mode).

        Charges are estimated per book at the limit price: the buy-side charges for a BUY, the
        sell-side charges (with the DP charge) for a SELL.
        """
        side = Side(idea.side)
        label = "buy" if side is Side.BUY else "sell"
        lines = [f"{side.value} {idea.symbol}"]
        for book, qty in card.sizes.items():
            if qty <= 0:
                continue
            lines.append(f"{book}: {qty} {'unit' if qty == 1 else 'units'}, "
                         f"est. {label} charges {self._charges_text(side, qty, idea.limit_price, charges_cfg)}")
        lines += [
            f"Limit: {_rupees(idea.limit_price)}",
            f"Target: {_rupees(idea.target)}",
            f"Stop: {_rupees(idea.stop)}",
            f"Reason: {idea.reason}",
            f"Expires: {card.expires_at.astimezone(IST).strftime('%H:%M')} IST",
        ]
        return "\n".join(lines)

    # ---- internals ----

    def _set_status(self, card: ApprovalCard, status: CardStatus) -> ApprovalCard:
        card = replace(card, status=status)
        self._store.save_card(card.id, _to_payload(card))
        return card

    def _price_confirmed(self, card: ApprovalCard, current_price: float) -> bool:
        if not _usable_price(current_price):
            log.warning("card %s: cannot confirm the current price (%r); closing it for a re-issue",
                        card.id, current_price)
            return False
        move = abs(current_price / card.quote_at_issue - 1)
        if move > self._cfg.reprice_tolerance_pct / 100 + _MOVE_EPSILON:
            log.info("card %s: price moved %.2f%% since issue (limit %.2f%%); closing it for a re-issue",
                     card.id, move * 100, self._cfg.reprice_tolerance_pct)
            return False
        return True

    @staticmethod
    def _charges_text(side: Side, qty: int, price: float, cfg: ChargesConfig) -> str:
        if not _usable_price(price):
            return "n/a"
        return _rupees(compute_charges(side, qty, price, cfg).total)
