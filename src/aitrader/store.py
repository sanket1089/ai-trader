"""SQLite persistence for the whole service (WAL mode).

Conventions:

* Datetimes are stored as ISO-8601 strings with a ``+05:30`` offset. Every datetime is
  converted to IST on write (naive datetimes are rejected), so the strings sort
  chronologically and range queries are plain string comparisons.
* On read, datetimes come back as aware IST datetimes and enum columns come back as real
  ``Side`` / ``OrderStatus`` members. ``charges.compute_charges`` checks ``side is Side.BUY``,
  so a plain string would silently undercharge.
* Dates (equity snapshots, daily counters) are ``datetime.date``, never ``datetime``.
"""

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from aitrader.calendar import IST
from aitrader.models import Fill, Order, OrderStatus, Position, Side, TradeIdea

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id          TEXT PRIMARY KEY,
    book        TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    qty         INTEGER NOT NULL,
    limit_price REAL NOT NULL,
    reason      TEXT NOT NULL,
    is_exit     INTEGER NOT NULL,
    idea_id     TEXT,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

CREATE TABLE IF NOT EXISTS fills (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    book     TEXT NOT NULL,
    symbol   TEXT NOT NULL,
    side     TEXT NOT NULL,
    qty      INTEGER NOT NULL,
    price    REAL NOT NULL,
    charges  REAL NOT NULL,
    is_exit  INTEGER NOT NULL,
    ts       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fills_book_ts ON fills(book, ts);

CREATE TABLE IF NOT EXISTS positions (
    book      TEXT NOT NULL,
    symbol    TEXT NOT NULL,
    qty       INTEGER NOT NULL,
    avg_price REAL NOT NULL,
    stop      REAL NOT NULL,
    target    REAL NOT NULL,
    opened_at TEXT NOT NULL,
    idea_id   TEXT,
    PRIMARY KEY (book, symbol)
);

CREATE TABLE IF NOT EXISTS cash (
    book  TEXT PRIMARY KEY,
    value REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ideas (
    id                TEXT PRIMARY KEY,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,
    limit_price       REAL NOT NULL,
    stop              REAL NOT NULL,
    target            REAL NOT NULL,
    expected_move_pct REAL NOT NULL,
    confidence        REAL NOT NULL,
    holding_days      INTEGER NOT NULL,
    reason            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cards (
    card_id TEXT PRIMARY KEY,
    status  TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cards_status ON cards(status);

CREATE TABLE IF NOT EXISTS risk_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    approved INTEGER NOT NULL,
    reasons  TEXT NOT NULL,
    ts       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_risk_log_order ON risk_log(order_id);

CREATE TABLE IF NOT EXISTS agent_outputs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    agent    TEXT NOT NULL,
    payload  TEXT NOT NULL,
    cost_usd REAL NOT NULL,
    ts       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read    INTEGER NOT NULL,
    cache_write   INTEGER NOT NULL,
    cost_usd      REAL NOT NULL,
    ts            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ai_usage_ts ON ai_usage(ts);

CREATE TABLE IF NOT EXISTS lessons (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    ts   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS equity (
    book   TEXT NOT NULL,
    day    TEXT NOT NULL,
    equity REAL NOT NULL,
    PRIMARY KEY (book, day)
);
"""


def _ts(dt: datetime) -> str:
    """Aware datetime -> fixed-width ISO string in IST."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("datetimes must be timezone-aware")
    return dt.astimezone(IST).isoformat(timespec="microseconds")


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(IST)


def _day(d: date) -> date:
    if isinstance(d, datetime):
        raise TypeError(f"expected a date, got a datetime: {d!r}")
    return d


def _day_bounds(d: date) -> tuple[str, str]:
    """[start, end) of the IST calendar day ``d`` as stored strings."""
    start = datetime.combine(_day(d), time.min, tzinfo=IST)
    return _ts(start), _ts(start + timedelta(days=1))


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    start = datetime(year, month, 1, tzinfo=IST)
    end = datetime(year + 1, 1, 1, tzinfo=IST) if month == 12 else datetime(year, month + 1, 1, tzinfo=IST)
    return _ts(start), _ts(end)


def _order(r: sqlite3.Row) -> Order:
    return Order(id=r["id"], book=r["book"], symbol=r["symbol"], side=Side(r["side"]), qty=r["qty"],
                 limit_price=r["limit_price"], reason=r["reason"], is_exit=bool(r["is_exit"]),
                 idea_id=r["idea_id"], created_at=_dt(r["created_at"]),
                 status=OrderStatus(r["status"]), attempts=r["attempts"])


def _fill(r: sqlite3.Row) -> Fill:
    return Fill(order_id=r["order_id"], book=r["book"], symbol=r["symbol"], side=Side(r["side"]),
                qty=r["qty"], price=r["price"], charges=r["charges"], is_exit=bool(r["is_exit"]),
                ts=_dt(r["ts"]))


def _position(r: sqlite3.Row) -> Position:
    return Position(book=r["book"], symbol=r["symbol"], qty=r["qty"], avg_price=r["avg_price"],
                    stop=r["stop"], target=r["target"], opened_at=_dt(r["opened_at"]),
                    idea_id=r["idea_id"])


def _idea(r: sqlite3.Row) -> TradeIdea:
    return TradeIdea(id=r["id"], symbol=r["symbol"], side=Side(r["side"]), limit_price=r["limit_price"],
                     stop=r["stop"], target=r["target"], expected_move_pct=r["expected_move_pct"],
                     confidence=r["confidence"], holding_days=r["holding_days"], reason=r["reason"])


class Store:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tx_depth = 0          # > 0 while inside transaction(); guarded by _lock
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group several writes into one atomic commit.

        Outside a transaction every write method commits on its own. Inside the block they
        don't: everything is committed once when the outermost block exits normally, and rolled
        back if it exits with an exception, so a crash can never leave a fill saved with its
        order still PENDING. The block is re-entrant: a nested ``transaction()`` joins the outer
        one, and an exception leaving any level rolls back the whole outer transaction. The store
        lock is held for the whole block, so other threads' reads and writes wait for it.
        """
        with self._lock:
            if self._tx_depth > 0:
                self._tx_depth += 1
                try:
                    yield
                finally:
                    self._tx_depth -= 1
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._tx_depth = 1
            try:
                yield
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            finally:
                self._tx_depth = 0

    # ---- helpers ----

    def _write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            if self._tx_depth > 0:          # the enclosing transaction() commits
                return self._conn.execute(sql, params)
            with self._conn:
                return self._conn.execute(sql, params)

    def _read(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _scalar(self, sql: str, params: tuple = ()) -> Any:
        return self._read(sql, params)[0][0]

    # ---- orders and fills ----

    def save_order(self, o: Order) -> None:
        self._write(
            "INSERT INTO orders (id, book, symbol, side, qty, limit_price, reason, is_exit, idea_id,"
            " created_at, status, attempts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (o.id, o.book, o.symbol, str(o.side), o.qty, o.limit_price, o.reason, int(o.is_exit),
             o.idea_id, _ts(o.created_at), str(o.status), o.attempts))

    def update_order(self, o: Order) -> None:
        cur = self._write(
            "UPDATE orders SET book=?, symbol=?, side=?, qty=?, limit_price=?, reason=?, is_exit=?,"
            " idea_id=?, created_at=?, status=?, attempts=? WHERE id=?",
            (o.book, o.symbol, str(o.side), o.qty, o.limit_price, o.reason, int(o.is_exit),
             o.idea_id, _ts(o.created_at), str(o.status), o.attempts, o.id))
        if cur.rowcount == 0:
            raise KeyError(f"no such order: {o.id}")

    def get_order(self, order_id: str) -> Order | None:
        """The order with this id in any status, or ``None``."""
        rows = self._read("SELECT * FROM orders WHERE id = ?", (order_id,))
        return _order(rows[0]) if rows else None

    def pending_orders(self, book: str | None = None) -> list[Order]:
        sql = "SELECT * FROM orders WHERE status = ?"
        params: tuple = (str(OrderStatus.PENDING),)
        if book is not None:
            sql += " AND book = ?"
            params += (book,)
        return [_order(r) for r in self._read(sql + " ORDER BY created_at, id", params)]

    def save_fill(self, f: Fill) -> None:
        self._write(
            "INSERT INTO fills (order_id, book, symbol, side, qty, price, charges, is_exit, ts)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (f.order_id, f.book, f.symbol, str(f.side), f.qty, f.price, f.charges, int(f.is_exit),
             _ts(f.ts)))

    def fills(self, book: str, start: datetime, end: datetime) -> list[Fill]:
        """Fills with ``start <= ts < end``, oldest first."""
        rows = self._read("SELECT * FROM fills WHERE book = ? AND ts >= ? AND ts < ? ORDER BY ts, id",
                          (book, _ts(start), _ts(end)))
        return [_fill(r) for r in rows]

    # ---- positions and cash ----

    def positions(self, book: str) -> list[Position]:
        rows = self._read("SELECT * FROM positions WHERE book = ? ORDER BY opened_at, symbol", (book,))
        return [_position(r) for r in rows]

    def upsert_position(self, p: Position) -> None:
        self._write(
            "INSERT OR REPLACE INTO positions (book, symbol, qty, avg_price, stop, target, opened_at,"
            " idea_id) VALUES (?,?,?,?,?,?,?,?)",
            (p.book, p.symbol, p.qty, p.avg_price, p.stop, p.target, _ts(p.opened_at), p.idea_id))

    def remove_position(self, book: str, symbol: str) -> None:
        self._write("DELETE FROM positions WHERE book = ? AND symbol = ?", (book, symbol))

    def has_cash(self, book: str) -> bool:
        """True once ``set_cash`` has seeded this book (used to detect a first run)."""
        return bool(self._read("SELECT 1 FROM cash WHERE book = ?", (book,)))

    def cash(self, book: str) -> float:
        """Cash of a seeded book. An unseeded book raises ``KeyError`` instead of reading as 0."""
        rows = self._read("SELECT value FROM cash WHERE book = ?", (book,))
        if not rows:
            raise KeyError(f"cash has not been set for book {book!r}")
        return rows[0]["value"]

    def set_cash(self, book: str, value: float) -> None:
        self._write("INSERT OR REPLACE INTO cash (book, value) VALUES (?, ?)", (book, value))

    # ---- state ----

    def get_state(self, key: str, default: Any = None) -> Any:
        rows = self._read("SELECT value FROM state WHERE key = ?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set_state(self, key: str, value: Any) -> None:
        self._write("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, json.dumps(value)))

    # ---- ideas and cards ----

    def save_idea(self, i: TradeIdea) -> None:
        self._write(
            "INSERT OR REPLACE INTO ideas (id, symbol, side, limit_price, stop, target, expected_move_pct,"
            " confidence, holding_days, reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (i.id, i.symbol, str(i.side), i.limit_price, i.stop, i.target, i.expected_move_pct,
             i.confidence, i.holding_days, i.reason))

    def get_idea(self, id: str) -> TradeIdea | None:
        rows = self._read("SELECT * FROM ideas WHERE id = ?", (id,))
        return _idea(rows[0]) if rows else None

    def save_card(self, card_id: str, payload: dict) -> None:
        """Upsert a card. ``payload["status"]`` (if present) is what ``cards_with_status`` filters on."""
        self._write(
            "INSERT INTO cards (card_id, status, payload) VALUES (?,?,?)"
            " ON CONFLICT(card_id) DO UPDATE SET status = excluded.status, payload = excluded.payload",
            (card_id, payload.get("status"), json.dumps(payload)))

    def get_card(self, card_id: str) -> dict | None:
        rows = self._read("SELECT payload FROM cards WHERE card_id = ?", (card_id,))
        return json.loads(rows[0]["payload"]) if rows else None

    def cards_with_status(self, status: str) -> list[dict]:
        """Card payloads with this status, in the order the cards were first saved."""
        rows = self._read("SELECT payload FROM cards WHERE status = ? ORDER BY rowid", (status,))
        return [json.loads(r["payload"]) for r in rows]

    # ---- risk audit ----

    def save_risk_decision(self, order_id: str, approved: bool, reasons: list[str], ts: datetime) -> None:
        self._write("INSERT INTO risk_log (order_id, approved, reasons, ts) VALUES (?,?,?,?)",
                    (order_id, int(approved), json.dumps(reasons), _ts(ts)))

    def risk_approved(self, order_id: str) -> bool:
        """True if the most recent risk decision for this order approved it. No decision means False."""
        rows = self._read("SELECT approved FROM risk_log WHERE order_id = ? ORDER BY id DESC LIMIT 1",
                          (order_id,))
        return bool(rows[0]["approved"]) if rows else False

    # ---- agents and AI usage ----

    def save_agent_output(self, agent: str, payload: str, cost_usd: float, ts: datetime) -> None:
        self._write("INSERT INTO agent_outputs (agent, payload, cost_usd, ts) VALUES (?,?,?,?)",
                    (agent, payload, cost_usd, _ts(ts)))

    def add_ai_usage(self, model: str, input_tokens: int, output_tokens: int, cache_read: int,
                     cache_write: int, cost_usd: float, ts: datetime) -> None:
        self._write(
            "INSERT INTO ai_usage (model, input_tokens, output_tokens, cache_read, cache_write, cost_usd, ts)"
            " VALUES (?,?,?,?,?,?,?)",
            (model, input_tokens, output_tokens, cache_read, cache_write, cost_usd, _ts(ts)))

    def _ai_spend_between(self, start: str, end: str) -> float:
        return float(self._scalar(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM ai_usage WHERE ts >= ? AND ts < ?", (start, end)))

    def ai_spend_usd(self, year: int, month: int) -> float:
        """Total AI spend in US$ for an IST calendar month."""
        return self._ai_spend_between(*_month_bounds(year, month))

    def ai_spend_usd_on(self, d: date) -> float:
        """Total AI spend in US$ for an IST calendar day."""
        return self._ai_spend_between(*_day_bounds(d))

    # ---- journal ----

    def add_lesson(self, text: str, ts: datetime) -> None:
        self._write("INSERT INTO lessons (text, ts) VALUES (?, ?)", (text, _ts(ts)))

    def recent_lessons(self, n: int) -> list[str]:
        """The ``n`` most recent lessons, newest first."""
        if n <= 0:
            return []
        rows = self._read("SELECT text FROM lessons ORDER BY ts DESC, id DESC LIMIT ?", (n,))
        return [r["text"] for r in rows]

    # ---- equity ----

    def save_equity_snapshot(self, book: str, d: date, equity: float) -> None:
        self._write("INSERT OR REPLACE INTO equity (book, day, equity) VALUES (?,?,?)",
                    (book, _day(d).isoformat(), equity))

    def equity_snapshot(self, book: str, d: date) -> float | None:
        rows = self._read("SELECT equity FROM equity WHERE book = ? AND day = ?", (book, _day(d).isoformat()))
        return rows[0]["equity"] if rows else None

    def latest_equity_snapshot_before(self, book: str, d: date) -> float | None:
        """Equity of the most recent snapshot dated strictly before ``d``, or ``None`` if there is none."""
        # days are stored as ISO strings (YYYY-MM-DD), which sort chronologically
        rows = self._read("SELECT equity FROM equity WHERE book = ? AND day < ? ORDER BY day DESC LIMIT 1",
                          (book, _day(d).isoformat()))
        return rows[0]["equity"] if rows else None

    # ---- counters (days are IST calendar days) ----

    def entries_on(self, book: str, d: date) -> int:
        """Number of non-exit BUY fills on day ``d``."""
        start, end = _day_bounds(d)
        return int(self._scalar(
            "SELECT COUNT(*) FROM fills WHERE book = ? AND side = ? AND is_exit = 0 AND ts >= ? AND ts < ?",
            (book, str(Side.BUY), start, end)))

    def bought_today(self, book: str, symbol: str, d: date) -> bool:
        """True if any BUY of ``symbol`` filled on day ``d``."""
        start, end = _day_bounds(d)
        return bool(self._read(
            "SELECT 1 FROM fills WHERE book = ? AND symbol = ? AND side = ? AND ts >= ? AND ts < ? LIMIT 1",
            (book, symbol, str(Side.BUY), start, end)))

    def sell_days_in_week(self, book: str, d: date) -> int:
        """Distinct days, from the Monday of ``d``'s week through ``d``, with a non-exit SELL fill."""
        d = _day(d)
        start, _ = _day_bounds(d - timedelta(days=d.weekday()))
        _, end = _day_bounds(d)
        # ts is always stored in IST, so its first 10 characters are the IST date.
        return int(self._scalar(
            "SELECT COUNT(DISTINCT substr(ts, 1, 10)) FROM fills"
            " WHERE book = ? AND side = ? AND is_exit = 0 AND ts >= ? AND ts < ?",
            (book, str(Side.SELL), start, end)))
