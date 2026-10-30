import logging
import sqlite3
import threading
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from exec_fakes import FakeMarketData, FakeNotifier
from pydantic import SecretStr
from telegram import InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram._utils.defaultvalue import DefaultValue
from telegram.ext import CallbackContext, CallbackQueryHandler, CommandHandler, MessageHandler

from aitrader.approvals import ApprovalBook, get_autopilot
from aitrader.calendar import at
from aitrader.config import load_settings
from aitrader.control import COMMAND_NAMES
from aitrader.models import Order, Position, Side, TradeIdea
from aitrader.risk.limits import LimitState, LossLimits
from aitrader.store import Store
from aitrader.telegram_bot import TelegramNotifier, _chunks, build_application

OWNER = 424242
STRANGER = 999
TOKEN = "123456:ABC-secret-token-value"
REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
SETTINGS = load_settings(REPO_CONFIG, env={"TELEGRAM_OWNER_CHAT_ID": str(OWNER), "TELEGRAM_BOT_TOKEN": TOKEN})

MON = date(2026, 11, 2)
NOW = at(MON, "10:00")
SYMBOL = "NIFTYBEES"
REAL, LEARN = "realistic", "learning"


class FakeBot:
    """Records every call PTB objects make on their bot; nothing touches the network."""

    username = "ai_trader_test_bot"       # a real bot knows its name after initialize()

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.edit_error: Exception | None = None

    async def answer_callback_query(self, *args, **kwargs):
        self.calls.append(("answer", kwargs))

    async def edit_message_text(self, *args, **kwargs):
        if self.edit_error is not None:
            raise self.edit_error
        self.calls.append(("edit", kwargs))

    async def send_message(self, *args, **kwargs):
        self.calls.append(("send", kwargs))

    def of(self, kind: str) -> list[dict]:
        return [kw for k, kw in self.calls if k == kind]


class SpyExecutor:
    def __init__(self):
        self.calls: list[Order] = []
        self.cancelled: list[str] = []

    async def submit(self, order, quote, now, idea=None):
        self.calls.append(order)
        return None

    def cancel_order(self, order_id, why):
        self.cancelled.append(order_id)
        return True


@pytest.fixture
def bot():
    return FakeBot()


@pytest.fixture
def svc(tmp_path):
    store = Store(tmp_path / "t.db")
    store.set_cash(REAL, 1000.0)
    store.set_cash(LEARN, 100000.0)
    market_data = FakeMarketData()
    market_data.set(SYMBOL, 280.0, NOW - timedelta(minutes=1))
    settings = SETTINGS.model_copy(update={"telegram_bot_token": SecretStr(TOKEN)})
    yield SimpleNamespace(
        store=store, settings=settings, executor=SpyExecutor(), market_data=market_data,
        limits={b.name: LossLimits(store, b, settings.holidays) for b in settings.books},
        approvals=ApprovalBook(store, settings.approval, settings.risk.entry_end),
        notifier=FakeNotifier(), llm=SimpleNamespace())
    store.close()


@pytest.fixture
def app(svc):
    return build_application(svc, clock=lambda: NOW)


# ---------------------------------------------------------------- building PTB objects without a bot


def callback_update(bot, data, user_id=OWNER, chat_id=OWNER, text="BUY NIFTYBEES\nLimit: ₹281.00"):
    return Update.de_json({
        "update_id": 1,
        "callback_query": {
            "id": "cb1", "from": {"id": user_id, "is_bot": False, "first_name": "X"},
            "chat_instance": "ci", "data": data,
            "message": {"message_id": 7, "date": 1700000000, "chat": {"id": chat_id, "type": "private"},
                        "text": text}}}, bot)


def message_update(bot, text, chat_id=OWNER, user_id=None):
    """A message in ``chat_id`` sent by ``user_id`` (default: the chat's own user, as in a private chat)."""
    entities = []
    if text.startswith("/"):
        entities = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
    return Update.de_json({
        "update_id": 2,
        "message": {"message_id": 1, "date": 1700000000, "chat": {"id": chat_id, "type": "private"},
                    "from": {"id": chat_id if user_id is None else user_id, "is_bot": False,
                             "first_name": "X"}, "text": text, "entities": entities}}, bot)


async def dispatch(app, update):
    """What ``Application.process_update`` does (minus the initialisation that would call Telegram):
    the first matching handler of each group runs."""
    for group in sorted(app.handlers):
        for handler in app.handlers[group]:
            check = handler.check_update(update)
            if check is None or check is False:
                continue
            context = CallbackContext.from_update(update, app)
            await handler.handle_update(update, app, check, context)
            break


def parse_mode(kwargs):
    """The parse mode a PTB call carries: PTB passes its "use the bot's default" marker, which is None
    here because the application sets no defaults."""
    return DefaultValue.get_value(kwargs.get("parse_mode"))


def make_idea(limit=281.0):
    return TradeIdea(id="i1", symbol=SYMBOL, side=Side.BUY, limit_price=limit, stop=270.0, target=300.0,
                     expected_move_pct=8.0, confidence=0.7, holding_days=5, reason="trend is up")


def new_card(svc):
    return svc.approvals.create(make_idea(), {REAL: 3, LEARN: 71}, 280.0, NOW)


def hard_stop(svc):
    assert svc.limits[REAL].evaluate(400.0, NOW).state is LimitState.HARD_STOPPED


# ---------------------------------------------------------------- registration


def test_every_command_is_registered_for_the_owner_chat_only(app):
    handlers = [h for hs in app.handlers.values() for h in hs]
    commands = [h for h in handlers if isinstance(h, CommandHandler)]
    assert sorted(c for h in commands for c in h.commands) == sorted(COMMAND_NAMES)
    assert {"status", "autopilot", "positions", "pnl", "journal", "pause", "resume", "stop", "budget",
            "help"} <= set(COMMAND_NAMES)
    for h in commands:
        assert h.filters.chat_ids == {OWNER}
    callbacks = [h for h in handlers if isinstance(h, CallbackQueryHandler)]
    assert [h.pattern.pattern for h in callbacks if h.pattern] == [r"^(approve|reject):", r"^recovery:"]
    assert sum(isinstance(h, MessageHandler) for h in handlers) == 1       # the stranger catch-all


@pytest.mark.parametrize("owner", [0, -1, -1001234567890])
def test_a_non_positive_owner_id_is_refused_because_groups_are_not_supported(svc, owner):
    svc.settings = svc.settings.model_copy(update={"telegram_owner_chat_id": owner})
    with pytest.raises(ValueError, match="group chats are not supported"):
        build_application(svc)


def test_httpx_request_logs_are_kept_quiet_because_their_url_holds_the_token(app):
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


# ---------------------------------------------------------------- strangers get nothing


async def test_non_owner_callback_ignored(app, bot, svc, caplog):
    card = new_card(svc)
    with caplog.at_level(logging.INFO):
        await dispatch(app, callback_update(bot, f"approve:{card.id}", user_id=STRANGER, chat_id=STRANGER))

    assert svc.executor.calls == []                                        # nothing was ordered
    assert svc.approvals.get(card.id).status == "open"                     # and the card was not touched
    assert bot.of("edit") == [] and bot.of("send") == []
    [answer] = bot.of("answer")
    assert answer.get("text") is None                                      # answered with no text
    assert str(STRANGER) in caplog.text
    assert TOKEN not in caplog.text


async def test_non_owner_user_in_the_owners_chat_is_ignored_too(app, bot, svc):
    card = new_card(svc)
    await dispatch(app, callback_update(bot, f"approve:{card.id}", user_id=STRANGER, chat_id=OWNER))
    assert svc.executor.calls == [] and svc.approvals.get(card.id).status == "open"
    assert bot.of("edit") == []


async def test_non_owner_cannot_approve_a_recovery_plan(app, bot, svc):
    hard_stop(svc)
    await dispatch(app, callback_update(bot, "recovery:realistic:0", user_id=STRANGER, chat_id=STRANGER))
    assert svc.store.get_state("recovery_approved:realistic:0") is None
    assert bot.of("edit") == [] and bot.of("send") == []


async def test_non_owner_callback_with_unknown_data_is_answered_silently(app, bot, svc):
    await dispatch(app, callback_update(bot, "something:else", user_id=STRANGER, chat_id=STRANGER))
    [answer] = bot.of("answer")
    assert answer.get("text") is None
    assert bot.of("edit") == [] and bot.of("send") == []


@pytest.mark.parametrize("text", ["/stop", "/autopilot on", "/pause", "/status"])
async def test_a_command_in_the_owners_chat_from_another_user_gets_no_reply_and_does_nothing(
        app, bot, svc, caplog, text):
    with caplog.at_level(logging.INFO):
        await dispatch(app, message_update(bot, text, chat_id=OWNER, user_id=STRANGER))

    assert bot.calls == []
    assert svc.store.get_state("manual_stop") is None and svc.store.get_state("paused") is None
    assert get_autopilot(svc.store) is False
    assert f"user id {STRANGER}" in caplog.text


async def test_non_owner_command_gets_no_reply_and_changes_nothing(app, bot, svc, caplog):
    with caplog.at_level(logging.INFO):
        await dispatch(app, message_update(bot, "/stop", chat_id=STRANGER))
        await dispatch(app, message_update(bot, "hello there", chat_id=STRANGER))

    assert bot.calls == []                                                 # no reply of any kind
    assert svc.store.get_state("manual_stop") is None
    assert caplog.text.count(f"non-owner chat id {STRANGER}") == 2
    assert "hello there" not in caplog.text                                # ids are logged, not content


# ---------------------------------------------------------------- the owner's commands


async def test_owner_command_gets_a_plain_text_reply(app, bot):
    await dispatch(app, message_update(bot, "/status"))

    [sent] = bot.of("send")
    assert sent["chat_id"] == OWNER
    assert "Autopilot: OFF" in sent["text"] and "realistic: ACTIVE" in sent["text"]
    assert parse_mode(sent) is None                                        # plain text, no markup parsing


async def test_owner_command_arguments_reach_the_handler(app, bot, svc):
    await dispatch(app, message_update(bot, "/autopilot on"))
    assert get_autopilot(svc.store) is True
    assert "Autopilot ON" in bot.of("send")[0]["text"]


async def test_owner_stop_command_sets_the_kill_switch(app, bot, svc):
    svc.store.save_order(Order(id="e1", book=REAL, symbol=SYMBOL, side=Side.BUY, qty=1, limit_price=280.0,
                               reason="trend", is_exit=False, idea_id="i1", created_at=NOW))

    await dispatch(app, message_update(bot, "/stop"))

    assert svc.store.get_state("manual_stop") is True
    assert svc.executor.cancelled == ["e1"]
    assert "STOP" in bot.of("send")[0]["text"]


async def test_owner_resume_fetches_its_quotes_in_a_thread(app, bot, svc):
    hard_stop(svc)
    svc.store.upsert_position(Position(book=REAL, symbol=SYMBOL, qty=1, avg_price=280.0, stop=270.0,
                                       target=300.0, opened_at=NOW - timedelta(days=1), idea_id="i1"))
    svc.store.set_state("recovery_approved:realistic:0", True)
    loop_thread = threading.get_ident()
    threads: list[int] = []
    real_quotes = svc.market_data.quotes

    def quotes(symbols):
        threads.append(threading.get_ident())
        return real_quotes(symbols)

    svc.market_data.quotes = quotes
    await dispatch(app, message_update(bot, "/resume"))

    assert threads and all(t != loop_thread for t in threads)
    assert "hard stop lifted" in bot.of("send")[0]["text"]


async def test_long_replies_are_split_under_the_telegram_limit(app, bot, svc):
    for i in range(20):
        svc.store.add_lesson(f"lesson {i} " + "x" * 480, NOW + timedelta(minutes=i))

    await dispatch(app, message_update(bot, "/journal 20"))

    sent = bot.of("send")
    assert len(sent) > 1
    assert all(len(kw["text"]) <= 4096 for kw in sent)
    assert "lesson 19" in sent[0]["text"] and "lesson 0 " in sent[-1]["text"]


# ---------------------------------------------------------------- card taps


async def test_owner_approve_places_orders_and_edits_the_card_message(app, bot, svc):
    card = new_card(svc)

    await dispatch(app, callback_update(bot, f"approve:{card.id}"))

    assert [(o.book, o.qty) for o in svc.executor.calls] == [(REAL, 3), (LEARN, 71)]
    assert svc.approvals.get(card.id).status == "approved"
    assert [k for k, _ in bot.calls] == ["answer", "edit"]                 # spinner stopped first
    [edit] = bot.of("edit")
    assert edit["chat_id"] == OWNER and edit["message_id"] == 7
    assert edit["text"].startswith("BUY NIFTYBEES\nLimit: ₹281.00\n\nApproved BUY NIFTYBEES.")
    assert edit.get("reply_markup") is None                                # no reply_markup: keyboard removed
    assert parse_mode(edit) is None


async def test_owner_reject_edits_the_card_message(app, bot, svc):
    card = new_card(svc)

    await dispatch(app, callback_update(bot, f"reject:{card.id}"))

    assert svc.executor.calls == []
    assert svc.approvals.get(card.id).status == "rejected"
    [edit] = bot.of("edit")
    assert edit["text"].endswith("Rejected BUY NIFTYBEES. Nothing was ordered.")
    assert edit.get("reply_markup") is None


async def test_card_tap_that_blows_up_is_reported_without_the_error_text(app, bot, svc, caplog):
    def broken(card_id):
        raise RuntimeError(f"boom {TOKEN}")

    svc.approvals = SimpleNamespace(get=broken)
    with caplog.at_level(logging.INFO):
        await dispatch(app, callback_update(bot, "approve:abc"))

    [sent] = bot.of("send")                                                # a new message: the card may still be
    assert bot.of("edit") == []                                            # open, so its buttons are left alone
    assert sent["chat_id"] == OWNER
    assert "Could not process that tap (RuntimeError)" in sent["text"]
    assert TOKEN not in sent["text"] and TOKEN not in caplog.text
    assert "RuntimeError" in caplog.text


async def test_no_price_reply_keeps_the_card_buttons_and_a_second_tap_later_approves(app, bot, svc):
    card = new_card(svc)
    svc.market_data.by_symbol.clear()                                      # the feed has no price right now

    await dispatch(app, callback_update(bot, f"approve:{card.id}"))

    assert bot.of("edit") == []                  # the card message is untouched, so its ✅/❌ keyboard stays
    [notice] = bot.of("send")
    assert notice["chat_id"] == OWNER
    assert "No current price" in notice["text"] and "tap again" in notice["text"]
    assert svc.approvals.get(card.id).status == "open"
    assert svc.executor.calls == []

    svc.market_data.set(SYMBOL, 280.0, NOW - timedelta(minutes=1))         # the price is back: tap again
    await dispatch(app, callback_update(bot, f"approve:{card.id}"))

    assert svc.approvals.get(card.id).status == "approved"
    assert [(o.book, o.qty) for o in svc.executor.calls] == [(REAL, 3), (LEARN, 71)]
    [edit] = bot.of("edit")                                                # now it is resolved: keyboard goes
    assert "Approved BUY NIFTYBEES." in edit["text"] and edit.get("reply_markup") is None


async def test_missing_idea_reply_keeps_the_card_buttons(app, bot, svc, tmp_path):
    card = new_card(svc)
    with closing(sqlite3.connect(tmp_path / "t.db")) as conn, conn:
        conn.execute("DELETE FROM ideas WHERE id = 'i1'")

    await dispatch(app, callback_update(bot, f"approve:{card.id}"))

    assert bot.of("edit") == []
    assert "idea behind this card is missing" in bot.of("send")[0]["text"]
    assert svc.approvals.get(card.id).status == "open"


async def test_a_resolved_card_always_loses_its_keyboard(app, bot, svc):
    card = new_card(svc)
    await dispatch(app, callback_update(bot, f"reject:{card.id}"))
    await dispatch(app, callback_update(bot, f"approve:{card.id}"))        # a late second tap: already handled
    assert len(bot.of("edit")) == 2 and bot.of("send") == []
    assert "already handled" in bot.of("edit")[1]["text"]


async def test_if_the_card_message_cannot_be_edited_the_outcome_is_sent_as_a_new_message(app, bot, svc):
    card = new_card(svc)
    bot.edit_error = BadRequest("Message to edit not found")

    await dispatch(app, callback_update(bot, f"reject:{card.id}"))

    [sent] = bot.of("send")
    assert sent["chat_id"] == OWNER and sent["text"] == "Rejected BUY NIFTYBEES. Nothing was ordered."


# ---------------------------------------------------------------- recovery taps


async def test_owner_recovery_tap_approves_the_current_stop(app, bot, svc):
    hard_stop(svc)

    await dispatch(app, callback_update(bot, "recovery:realistic:0", text="Recovery plan for realistic"))

    assert svc.store.get_state("recovery_approved:realistic:0") is True
    [edit] = bot.of("edit")
    assert edit["text"] == ("Recovery plan for realistic\n\n"
                            "Recovery plan approved — send /resume to restart realistic")
    assert edit.get("reply_markup") is None


async def test_recovery_tap_for_another_stop_changes_nothing(app, bot, svc):
    hard_stop(svc)

    await dispatch(app, callback_update(bot, "recovery:realistic:5"))

    assert svc.store.get_state("recovery_approved:realistic:5") is None
    assert svc.store.get_state("recovery_approved:realistic:0") is None
    assert "out of date" in bot.of("edit")[0]["text"]


@pytest.mark.parametrize("data", ["recovery:", "recovery:realistic", "recovery:realistic:x", "recovery::3"])
async def test_malformed_recovery_buttons_change_nothing(app, bot, svc, data):
    hard_stop(svc)
    await dispatch(app, callback_update(bot, data))
    assert svc.store.get_state("recovery_approved:realistic:0") is None
    assert "Unrecognised recovery button" in bot.of("edit")[0]["text"]


# ---------------------------------------------------------------- errors and the token


async def test_error_handler_logs_only_the_error_type_and_place(app, caplog):
    [handler] = app.error_handlers

    def explode():
        raise RuntimeError(f"url with {TOKEN}")

    try:
        explode()
    except RuntimeError as exc:
        error = exc
    with caplog.at_level(logging.INFO):
        await handler(None, SimpleNamespace(error=error))

    assert "RuntimeError" in caplog.text and "test_telegram_bot.py" in caplog.text
    assert TOKEN not in caplog.text


# ---------------------------------------------------------------- the notifier


async def test_send_card_attaches_approve_and_reject_buttons(bot, svc):
    card = new_card(svc)

    await TelegramNotifier(bot, OWNER).send_card(card, "BUY NIFTYBEES")

    [sent] = bot.of("send")
    assert sent["chat_id"] == OWNER and sent["text"] == "BUY NIFTYBEES"
    assert parse_mode(sent) is None
    markup = sent["reply_markup"]
    assert isinstance(markup, InlineKeyboardMarkup)
    [row] = markup.inline_keyboard
    assert [(b.text, b.callback_data) for b in row] == [("✅ Approve", f"approve:{card.id}"),
                                                         ("❌ Reject", f"reject:{card.id}")]
    assert all(len(b.callback_data.encode()) <= 64 for b in row)           # Telegram's callback data limit


async def test_send_is_plain_text_and_splits_long_messages(bot):
    notifier = TelegramNotifier(bot, OWNER)
    await notifier.send("hello")
    await notifier.send("\n".join(["y" * 100] * 100))

    sent = bot.of("send")
    assert sent[0] == {"chat_id": OWNER, "text": "hello"}
    assert len(sent) == 1 + 3                                              # 10,099 characters: three pieces
    assert all(len(kw["text"]) <= 4096 for kw in sent)
    assert all("reply_markup" not in kw and "parse_mode" not in kw for kw in sent)


async def test_send_recovery_puts_the_button_under_the_last_piece(bot):
    notifier = TelegramNotifier(bot, OWNER)
    await notifier.send_recovery(REAL, 3, "\n".join(["plan line " * 10] * 80))

    sent = bot.of("send")
    assert len(sent) > 1
    assert all("reply_markup" not in kw for kw in sent[:-1])
    [row] = sent[-1]["reply_markup"].inline_keyboard
    assert [b.callback_data for b in row] == ["recovery:realistic:3"]


async def test_notifier_lets_send_errors_propagate_for_the_executor_to_handle(svc):
    class DownBot:
        async def send_message(self, **kwargs):
            raise RuntimeError("telegram is down")

    with pytest.raises(RuntimeError):
        await TelegramNotifier(DownBot(), OWNER).send("hi")


# ---------------------------------------------------------------- text splitting


def test_short_text_is_one_chunk():
    assert _chunks("a\nb") == ["a\nb"]


def test_chunks_break_on_lines_and_respect_the_limit():
    text = "\n".join(f"line {i:03d}" for i in range(100))
    pieces = _chunks(text, limit=100)
    assert all(len(p) <= 100 for p in pieces)
    assert "\n".join(pieces) == text                                       # nothing lost, split at line breaks


def test_one_enormous_line_is_cut():
    pieces = _chunks("x" * 250, limit=100)
    assert [len(p) for p in pieces] == [100, 100, 50]


# ---------------------------------------------------------------- the token never leaks


async def test_the_bot_token_is_never_logged(app, bot, svc, caplog):
    card = new_card(svc)
    with caplog.at_level(logging.DEBUG):
        await dispatch(app, message_update(bot, "/status"))
        await dispatch(app, message_update(bot, "/stop", chat_id=STRANGER))
        await dispatch(app, callback_update(bot, f"approve:{card.id}"))
        await dispatch(app, callback_update(bot, "approve:x", user_id=STRANGER, chat_id=STRANGER))
    assert TOKEN not in caplog.text
    assert TOKEN not in "".join((kw.get("text") or "") for _, kw in bot.calls)
