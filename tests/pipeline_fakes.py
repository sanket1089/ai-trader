"""Fakes and a Services builder shared by the pipeline, go-live and app tests. All offline."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from aitrader.app import build_services
from aitrader.config import load_settings
from aitrader.models import Quote

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
SETTINGS = load_settings(REPO_CONFIG, env={"TELEGRAM_OWNER_CHAT_ID": "42", "TELEGRAM_BOT_TOKEN": "x"})
REAL, LEARN = "realistic", "learning"

NIFTY_CSV = ("Company Name,Industry,Symbol,Series,ISIN Code\n"
             "Reliance Industries Ltd.,Oil,RELIANCE,EQ,INE002A01018\n"
             "ITC Ltd.,FMCG,ITC,EQ,INE154A01025\n")


class FakeMarketData:
    """Quotes at ``prices`` stamped ``ts``; 60 rows of daily history ending at the price."""

    def __init__(self, prices: dict[str, float], ts: datetime, history: bool = True):
        self.prices = dict(prices)
        self.ts = ts
        self.history = history
        self.funds: dict[str, dict] = {}
        self.history_calls: list[str] = []

    def quotes(self, symbols):
        return {s: Quote(s, self.prices[s], self.ts) for s in symbols if s in self.prices}

    def quote(self, symbol):
        return self.quotes([symbol]).get(symbol)

    def daily_history(self, symbol, days):
        self.history_calls.append(symbol)
        if not self.history or symbol not in self.prices:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        last = self.prices[symbol]
        closes = [last * (1 - 0.001 * (days - 1 - i)) for i in range(days)]
        return pd.DataFrame({"open": closes, "high": [c * 1.01 for c in closes],
                             "low": [c * 0.99 for c in closes], "close": closes,
                             "volume": [1000.0] * days})

    def fundamentals(self, symbol):
        return self.funds.get(symbol, {})

    def index_snapshot(self):
        return {"nifty_pct": 0.2, "banknifty_pct": 0.1, "fii_net_cr": None, "dii_net_cr": None}


class FakeLLM:
    """Returns canned outputs keyed by the output model's class name; records every call."""

    def __init__(self, responses: dict | None = None):
        self.responses = dict(responses or {})
        self.calls: list[tuple[str, str]] = []          # (agent, output model name)

    def run(self, agent, system, user, output_model, max_tokens=16000):
        self.calls.append((agent, output_model.__name__))
        return self.responses.get(output_model.__name__)


class RecordingNotifier:
    def __init__(self):
        self.texts: list[str] = []
        self.cards: list[tuple[object, str]] = []
        self.recoveries: list[tuple[str, int, str]] = []

    async def send(self, text):
        self.texts.append(text)

    async def send_card(self, card, text):
        self.cards.append((card, text))

    async def send_recovery(self, book, stop_id, text):
        self.recoveries.append((book, stop_id, text))


class SpyExecutor:
    def __init__(self):
        self.submits: list[tuple] = []
        self.retries: list[tuple] = []

    async def submit(self, order, quote, now, idea=None):
        self.submits.append((order, quote, now, idea))
        return None

    async def retry_pending(self, quotes, now):
        self.retries.append((dict(quotes), now))
        return []


class SpyExits:
    def __init__(self):
        self.checks: list[datetime] = []

    async def check(self, now):
        self.checks.append(now)
        return []


class FakeResp:
    def __init__(self, text="", status=200):
        self.text = text
        self.content = text.encode()
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def csv_get(text=NIFTY_CSV):
    def get(url, **kw):
        return FakeResp(text)
    return get


def failing_get(url, **kw):
    raise ConnectionError("network down https://secret")


def make_svc(tmp_path, md, llm=None, notifier=None, *, spy=True, feeds=(), now=None, http_get=None):
    """Real Services (real Store in ``tmp_path``) around fakes. ``spy`` swaps in SpyExecutor/SpyExits."""
    settings = SETTINGS.model_copy(update={
        "db_path": tmp_path / "data" / "aitrader.db",
        "data": SETTINGS.data.model_copy(update={"rss_feeds": list(feeds)}),
    })
    svc = build_services(settings, notifier or RecordingNotifier(), dry_run=False, market_data=md,
                         llm_client=llm or FakeLLM(), http_get=http_get or csv_get(),
                         clock=(lambda: now) if now is not None else None)
    if spy:
        svc.executor = SpyExecutor()
        svc.exits = SpyExits()
    svc.fetch_closes = lambda d: {}
    svc.news_parse = lambda url: {"entries": [], "bozo": 0}
    return svc


def minutes_before(t: datetime, m: int) -> datetime:
    return t - timedelta(minutes=m)
