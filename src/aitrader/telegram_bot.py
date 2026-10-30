"""Telegram adapter (python-telegram-bot v21+): carries text between the owner and ``control.py``.

* Only the owner is served. Every handler checks the chat id (commands) or the user id (taps) against
  ``settings.telegram_owner_chat_id``. Anyone else is logged by id only and gets no reply: a
  stranger's callback query is answered with no text, so the button stops spinning, and nothing else.
* Messages are plain text (no ``parse_mode``): lesson and idea text comes from a model and from news.
* Secrets: the bot token is never logged. PTB's HTTP client (httpx) logs every request URL at INFO,
  and the URL holds the token, so ``build_application`` raises the httpx loggers to WARNING. The error
  handler logs only the error's type and place, never its message.
* Cards: ``TelegramNotifier.send_card`` attaches the ✅ ``approve:<id>`` and ❌ ``reject:<id>``
  buttons. When a tap resolves the card, its message is edited to append the outcome and the keyboard
  is removed (an edit without ``reply_markup`` drops the keyboard).
* Updates are handled one at a time (PTB's default), so two taps on one card never run together.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from aitrader.approvals import ApprovalCard
from aitrader.calendar import now_ist
from aitrader.control import COMMAND_NAMES, handle_card, handle_command, handle_recovery
from aitrader.execution import describe_error

log = logging.getLogger(__name__)

TEXT_LIMIT = 4000        # Telegram refuses a text above 4096 characters

# Commands that read quotes from the (blocking) market data feed run in a thread so they do not
# freeze the bot. The ones that change state (/stop, /pause, /resume, /autopilot) run inline: with no
# await in them, nothing else on the event loop can place an order halfway through.
_THREADED = frozenset({"status", "positions", "pnl"})

_CARD_PATTERN = r"^(approve|reject):"
_RECOVERY_PATTERN = r"^recovery:"


def _chunks(text: str, limit: int = TEXT_LIMIT) -> list[str]:
    """``text`` cut into pieces of at most ``limit`` characters, at line breaks where possible."""
    pieces: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:                                  # one enormous line: cut it
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:limit])
            line = line[limit:]
        if current and len(current) + 1 + len(line) > limit:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    pieces.append(current)
    return pieces


def _clip(text: str, limit: int = TEXT_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


class TelegramNotifier:
    """The owner's chat as a ``Notifier``. Send errors propagate; the Executor logs and ignores them."""

    def __init__(self, bot: Any, chat_id: int):
        self._bot = bot
        self._chat_id = chat_id

    async def send(self, text: str) -> None:
        await self._send(text)

    async def send_card(self, card: ApprovalCard, text: str) -> None:
        markup = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Approve", callback_data=f"approve:{card.id}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"reject:{card.id}"),
        ]])
        await self._bot.send_message(chat_id=self._chat_id, text=_clip(text), reply_markup=markup)

    async def send_recovery(self, book: str, stop_id: int, text: str) -> None:
        """A recovery plan with one ✅ button, ``recovery:<book>:<stop_id>`` (see ``handle_recovery``)."""
        markup = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Approve recovery plan", callback_data=f"recovery:{book}:{stop_id}"),
        ]])
        await self._send(text, markup)

    async def _send(self, text: str, markup: InlineKeyboardMarkup | None = None) -> None:
        chunks = _chunks(text)
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            if markup is not None and last:                       # the keyboard goes under the last piece
                await self._bot.send_message(chat_id=self._chat_id, text=chunk, reply_markup=markup)
            else:
                await self._bot.send_message(chat_id=self._chat_id, text=chunk)


def build_application(svc: Any, clock: Callable[[], datetime] = now_ist) -> Application:
    """The PTB application for ``svc``. Building it makes no network call; ``run_polling`` does."""
    owner = svc.settings.telegram_owner_chat_id
    for noisy in ("httpx", "httpcore"):                           # their INFO lines carry the bot token
        logging.getLogger(noisy).setLevel(logging.WARNING)
    app = Application.builder().token(svc.settings.telegram_bot_token.get_secret_value()).build()

    def from_owner_chat(update: Update) -> bool:
        chat = update.effective_chat
        if chat is not None and chat.id == owner:
            return True
        log.warning("ignored a message from non-owner chat id %s", chat.id if chat is not None else None)
        return False

    async def owner_callback(update: Update) -> bool:
        """True for a tap from the owner. Anyone else's tap is logged and answered with no text."""
        query = update.callback_query
        user = query.from_user if query is not None else None
        if user is not None and user.id == owner:
            return True
        log.warning("ignored a callback query from non-owner id %s", user.id if user is not None else None)
        if query is not None:
            await _answer(query)
        return False

    def command(name: str):
        async def on_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not from_owner_chat(update) or update.effective_message is None:
                return
            args = list(context.args or [])
            if name in _THREADED:
                reply = await asyncio.to_thread(handle_command, name, args, svc, clock())
            else:
                reply = handle_command(name, args, svc, clock())
            for chunk in _chunks(reply):
                await update.effective_message.reply_text(chunk)
        return on_command

    async def on_stranger(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        from_owner_chat(update)                                   # logs the chat id; no reply

    async def on_card(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await owner_callback(update):
            return
        query = update.callback_query
        await _answer(query)
        action, _, card_id = (query.data or "").partition(":")
        try:
            outcome = await handle_card(card_id, action == "approve", svc, clock())
        except Exception as exc:  # noqa: BLE001 - the owner is told; only the type is logged
            log.error("handling a card tap failed: %s", describe_error(exc))
            outcome = f"Could not process that tap ({type(exc).__name__}); see the log."
        await _finish(query, outcome, owner)

    async def on_recovery(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await owner_callback(update):
            return
        query = update.callback_query
        await _answer(query)
        book, _, stop_id = (query.data or "").removeprefix("recovery:").rpartition(":")
        if not book or not stop_id.isdigit():
            outcome = "Unrecognised recovery button. Nothing changed."
        else:
            try:
                outcome = handle_recovery(book, int(stop_id), svc)
            except Exception as exc:  # noqa: BLE001
                log.error("handling a recovery tap failed: %s", describe_error(exc))
                outcome = f"Could not process that tap ({type(exc).__name__}); see the log."
        await _finish(query, outcome, owner)

    async def on_other_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await owner_callback(update):
            await _answer(update.callback_query)                  # not ours to act on: just stop the spinner

    async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        error = context.error
        log.error("telegram handler error: %s", describe_error(error) if error is not None else "unknown")

    owner_only = filters.Chat(chat_id=owner)
    for name in COMMAND_NAMES:
        app.add_handler(CommandHandler(name, command(name), filters=owner_only))
    app.add_handler(MessageHandler(~owner_only, on_stranger))
    app.add_handler(CallbackQueryHandler(on_card, pattern=_CARD_PATTERN))
    app.add_handler(CallbackQueryHandler(on_recovery, pattern=_RECOVERY_PATTERN))
    app.add_handler(CallbackQueryHandler(on_other_callback))
    app.add_error_handler(on_error)
    return app


async def _answer(query: Any) -> None:
    """Stop the button's spinner (no text). A failure is not worth failing the tap for."""
    try:
        await query.answer()
    except TelegramError as exc:
        log.warning("could not answer a callback query: %s", describe_error(exc))


async def _finish(query: Any, outcome: str, owner: int) -> None:
    """Append ``outcome`` to the card's message and drop the keyboard; if the message cannot be
    edited (too old, or unchanged), send the outcome as a new message instead."""
    original = getattr(query.message, "text", None) or ""
    try:
        await query.edit_message_text(text=_clip(f"{original}\n\n{outcome}" if original else outcome))
        return
    except TelegramError as exc:
        log.warning("could not edit the card message: %s", describe_error(exc))
    try:
        for chunk in _chunks(outcome):
            await query.get_bot().send_message(chat_id=owner, text=chunk)
    except TelegramError as exc:
        log.error("could not send the outcome either: %s", describe_error(exc))
