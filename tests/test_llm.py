import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from aitrader.agents import llm as llm_module
from aitrader.agents.llm import BudgetLevel, LLMClient, budget_level, cost_usd
from aitrader.calendar import IST
from aitrader.config import load_settings
from aitrader.store import Store

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
ENV = {"TELEGRAM_OWNER_CHAT_ID": "42", "TELEGRAM_BOT_TOKEN": "tok"}
NOW = datetime(2026, 11, 3, 10, 0, tzinfo=IST)
REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class Out(BaseModel):
    answer: str


@pytest.fixture
def ai_cfg():
    return load_settings(REPO_CONFIG, env=ENV).ai


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def usage(input_tokens=1000, output_tokens=200, cache_read=0, cache_write=0):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                           cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_write)


def response(parsed=None, stop_reason="end_turn", stop_details=None, use=None):
    return SimpleNamespace(parsed_output=parsed if parsed is not None else Out(answer="hi"),
                           stop_reason=stop_reason, stop_details=stop_details,
                           usage=use or usage())


class FakeClient:
    """Stands in for anthropic.Anthropic: exposes beta.messages.parse and records every call."""

    def __init__(self, result=None, exc=None):
        self.result = result if result is not None else response()
        self.exc = exc
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.result


def make(store, ai_cfg, fake):
    return LLMClient(store, ai_cfg, client=fake, clock=lambda: NOW)


def status_error(cls, code):
    return cls("boom", response=httpx.Response(code, request=REQUEST), body=None)


def seed_spend(store, usd, ts=NOW):
    store.add_ai_usage("claude-opus-5-5", 0, 0, 0, 0, usd, ts)


# ---- cost_usd ----

def test_cost_usd_opus(ai_cfg):
    u = usage(input_tokens=10_000, output_tokens=2_000, cache_read=50_000, cache_write=0)
    assert cost_usd(u, ai_cfg.prices["claude-opus-5-5"]) == pytest.approx(0.09)


def test_cost_usd_counts_cache_write_and_treats_none_as_zero(ai_cfg):
    price = ai_cfg.prices["claude-opus-5-5"]
    assert cost_usd(usage(0, 0, 0, 1_000_000), price) == pytest.approx(5.0)
    u = SimpleNamespace(input_tokens=1_000_000, output_tokens=0,
                        cache_read_input_tokens=None, cache_creation_input_tokens=None)
    assert cost_usd(u, price) == pytest.approx(4.0)


# ---- budget_level ----

@pytest.mark.parametrize("spend, level", [
    (0, BudgetLevel.NORMAL),
    (79.99, BudgetLevel.NORMAL),
    (80, BudgetLevel.REDUCED),
    (99.99, BudgetLevel.REDUCED),
    (100, BudgetLevel.EXHAUSTED),
    (250, BudgetLevel.EXHAUSTED),
])
def test_budget_level_thresholds(store, ai_cfg, spend, level):
    seed_spend(store, spend)
    assert budget_level(store, ai_cfg, NOW) == level


def test_budget_level_uses_ist_calendar_month(store, ai_cfg):
    # 2026-10-31 20:00 UTC is already 2026-11-01 01:30 IST: November's budget, not October's.
    seed_spend(store, 100, ts=datetime(2026, 10, 31, 12, 0, tzinfo=IST))   # October spend
    after_ist_midnight = datetime(2026, 10, 31, 20, 0, tzinfo=timezone.utc)
    assert budget_level(store, ai_cfg, after_ist_midnight) == BudgetLevel.NORMAL
    seed_spend(store, 100, ts=datetime(2026, 11, 1, 1, 0, tzinfo=IST))     # November spend
    assert budget_level(store, ai_cfg, after_ist_midnight) == BudgetLevel.EXHAUSTED


def test_budget_level_rejects_naive_datetime(store, ai_cfg):
    with pytest.raises(ValueError):
        budget_level(store, ai_cfg, datetime(2026, 11, 3, 10, 0))


# ---- run ----

def test_run_returns_parsed_and_records_usage(store, ai_cfg):
    fake = FakeClient(response(parsed=Out(answer="buy"), use=usage(10_000, 2_000, 50_000, 0)))
    result = make(store, ai_cfg, fake).run("news", "sys", "usr", Out)
    assert result == Out(answer="buy")
    assert store.ai_spend_usd(2026, 11) == pytest.approx(0.09)
    rows = store._read("SELECT * FROM ai_usage")
    assert len(rows) == 1
    r = rows[0]
    assert (r["model"], r["input_tokens"], r["output_tokens"], r["cache_read"], r["cache_write"]) == \
        ("claude-opus-5-5", 10_000, 2_000, 50_000, 0)
    outs = store._read("SELECT * FROM agent_outputs")
    assert len(outs) == 1
    assert outs[0]["agent"] == "news"
    assert json.loads(outs[0]["payload"]) == {"answer": "buy"}
    assert outs[0]["cost_usd"] == pytest.approx(0.09)


def test_run_passes_fallbacks_and_effort(store, ai_cfg):
    fake = FakeClient()
    make(store, ai_cfg, fake).run("news", "the system prompt", "the user prompt", Out, max_tokens=1234)
    (kw,) = fake.calls
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    assert kw["output_config"]["effort"] == "low"
    assert kw["model"] == "claude-opus-5-5"
    assert kw["max_tokens"] == 1234
    assert kw["output_format"] is Out
    assert kw["messages"] == [{"role": "user", "content": "the user prompt"}]
    assert kw["system"] == [{"type": "text", "text": "the system prompt",
                             "cache_control": {"type": "ephemeral"}}]
    # Opus 5.5 rejects these: thinking can't be disabled, no budget_tokens, no sampling params.
    assert not {"thinking", "temperature", "top_p", "top_k"} & kw.keys()


def test_run_uses_configured_effort_per_agent(store, ai_cfg):
    fake = FakeClient()
    make(store, ai_cfg, fake).run("portfolio_manager", "s", "u", Out)
    assert fake.calls[0]["output_config"]["effort"] == "medium"


def test_run_uses_the_agents_configured_model_and_its_price(store, ai_cfg):
    cfg = ai_cfg.model_copy(update={"agents": {
        **ai_cfg.agents, "news": ai_cfg.agents["news"].model_copy(update={"model": "claude-haiku-4-5"})}})
    fake = FakeClient(response(use=usage(1_000_000, 0)))
    make(store, cfg, fake).run("news", "s", "u", Out)
    assert fake.calls[0]["model"] == "claude-haiku-4-5"
    assert store.ai_spend_usd(2026, 11) == pytest.approx(1.0)       # haiku input: $1 / M


def test_reduced_budget_forces_low_effort(store, ai_cfg):
    seed_spend(store, 80)
    fake = FakeClient()
    assert make(store, ai_cfg, fake).run("portfolio_manager", "s", "u", Out) == Out(answer="hi")
    assert fake.calls[0]["output_config"]["effort"] == "low"


def test_exhausted_budget_skips_call(store, ai_cfg):
    seed_spend(store, 100)
    fake = FakeClient()
    assert make(store, ai_cfg, fake).run("portfolio_manager", "s", "u", Out) is None
    assert fake.calls == []
    assert len(store._read("SELECT * FROM ai_usage")) == 1          # only the seed row


def test_refusal_returns_none_but_records_usage(store, ai_cfg, caplog):
    details = SimpleNamespace(category="cyber", explanation=None)
    fake = FakeClient(response(parsed=Out(answer="must not be used"), stop_reason="refusal",
                               stop_details=details, use=usage(1_000_000, 0)))
    with caplog.at_level(logging.WARNING):
        assert make(store, ai_cfg, fake).run("news", "s", "u", Out) is None
    assert "refusal" in caplog.text and "cyber" in caplog.text
    assert store.ai_spend_usd(2026, 11) == pytest.approx(4.0)       # billed even though refused
    (out,) = store._read("SELECT * FROM agent_outputs")
    assert "must not be used" not in out["payload"]


def test_missing_parsed_output_returns_none(store, ai_cfg, caplog):
    fake = FakeClient()
    fake.result = SimpleNamespace(parsed_output=None, stop_reason="end_turn", stop_details=None,
                                  usage=usage())
    with caplog.at_level(logging.WARNING):
        assert make(store, ai_cfg, fake).run("news", "s", "u", Out) is None
    assert len(store._read("SELECT * FROM ai_usage")) == 1


@pytest.mark.parametrize("exc", [
    status_error(anthropic.RateLimitError, 429),
    anthropic.APIConnectionError(request=REQUEST),
    anthropic.APITimeoutError(request=REQUEST),
    status_error(anthropic.InternalServerError, 500),
    status_error(anthropic.APIStatusError, 529),
])
def test_transient_errors_return_none(store, ai_cfg, caplog, exc):
    fake = FakeClient(exc=exc)
    with caplog.at_level(logging.ERROR):
        assert make(store, ai_cfg, fake).run("news", "s", "u", Out) is None
    assert type(exc).__name__ in caplog.text
    assert store.ai_spend_usd(2026, 11) == 0


def test_rate_limit_returns_none(store, ai_cfg):
    fake = FakeClient(exc=status_error(anthropic.RateLimitError, 429))
    assert make(store, ai_cfg, fake).run("news", "s", "u", Out) is None


def test_bad_request_raises(store, ai_cfg):
    fake = FakeClient(exc=status_error(anthropic.BadRequestError, 400))
    with pytest.raises(anthropic.BadRequestError):
        make(store, ai_cfg, fake).run("news", "s", "u", Out)


@pytest.mark.parametrize("cls, code", [
    (anthropic.AuthenticationError, 401),
    (anthropic.PermissionDeniedError, 403),
    (anthropic.NotFoundError, 404),
])
def test_other_client_errors_raise(store, ai_cfg, cls, code):
    fake = FakeClient(exc=status_error(cls, code))
    with pytest.raises(cls):
        make(store, ai_cfg, fake).run("news", "s", "u", Out)


def test_unknown_agent_raises_clear_error(store, ai_cfg):
    fake = FakeClient()
    with pytest.raises(KeyError, match="no_such_agent"):
        make(store, ai_cfg, fake).run("no_such_agent", "s", "u", Out)
    assert fake.calls == []


def test_unknown_model_price_raises_clear_error(store, ai_cfg):
    cfg = ai_cfg.model_copy(update={"agents": {
        **ai_cfg.agents, "news": ai_cfg.agents["news"].model_copy(update={"model": "claude-mystery-1"})}})
    fake = FakeClient()
    with pytest.raises(KeyError, match="claude-mystery-1"):
        make(store, cfg, fake).run("news", "s", "u", Out)
    assert fake.calls == []


def test_config_bug_raises_even_when_budget_exhausted(store, ai_cfg):
    seed_spend(store, 100)
    with pytest.raises(KeyError, match="no_such_agent"):
        make(store, ai_cfg, FakeClient()).run("no_such_agent", "s", "u", Out)


def test_call_shape_is_accepted_by_the_real_sdk(store, ai_cfg):
    """Drive the real anthropic client over a mock transport (no network).

    The fakes above would happily accept a keyword the SDK rejects; this proves the installed
    SDK takes ``fallbacks``/``output_config``/``output_format`` and sends them on the wire.
    """
    httpx2 = pytest.importorskip("httpx2")      # the HTTP library the SDK itself uses
    seen = {}

    def handler(request):
        seen["beta"] = request.headers.get("anthropic-beta")
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": '{"answer": "from the wire"}'}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10_000, "output_tokens": 2_000,
                      "cache_read_input_tokens": 50_000, "cache_creation_input_tokens": None}})

    sdk = anthropic.Anthropic(api_key="dummy",
                              http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    result = make(store, ai_cfg, sdk).run("news", "sys", "usr", Out)

    assert result == Out(answer="from the wire")
    assert "server-side-fallback-2026-07-01" in seen["beta"]
    body = seen["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "low"
    assert "format" in body["output_config"]                 # output_format=Out became a JSON schema
    assert "thinking" not in body and "temperature" not in body
    assert store.ai_spend_usd(2026, 11) == pytest.approx(0.09)


# ---- smoke ----

SECRET_KEY = "sk-ant-test-secret-do-not-print"


@pytest.fixture
def repo_root(tmp_path, monkeypatch):
    shutil.copy(REPO_CONFIG, tmp_path / "config.yaml")
    # setenv-then-delenv makes monkeypatch restore the variable to "absent" even though
    # smoke() will set it through load_dotenv.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    return tmp_path


def test_smoke_loads_key_from_dotenv_next_to_config_and_never_prints_it(
        repo_root, monkeypatch, capsys, caplog):
    (repo_root / ".env").write_text(f"ANTHROPIC_API_KEY={SECRET_KEY}\n", encoding="utf-8")
    fake = FakeClient(response(parsed=llm_module._Ok(ok=True), use=usage(1000, 100)))
    key_seen = []

    def fake_anthropic():
        key_seen.append(os.environ.get("ANTHROPIC_API_KEY"))
        return fake

    monkeypatch.setattr(llm_module.anthropic, "Anthropic", fake_anthropic)
    with caplog.at_level(logging.DEBUG):
        llm_module.smoke(repo_root / "config.yaml")

    assert key_seen == [SECRET_KEY]                      # .env was loaded before the client was built
    out = capsys.readouterr()
    assert "ok=True" in out.out
    assert "cost=$0.0" in out.out
    assert SECRET_KEY not in out.out + out.err + caplog.text
    assert fake.calls[0]["model"] == "claude-opus-5-5"
    assert (repo_root / "data" / "aitrader.db").exists()   # db_path is resolved against the repo root


def test_smoke_without_a_key_exits_with_a_clear_message(repo_root, monkeypatch):
    monkeypatch.setattr(llm_module.anthropic, "Anthropic",
                        lambda: pytest.fail("must not build a client without a key"))
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        llm_module.smoke(repo_root / "config.yaml")
