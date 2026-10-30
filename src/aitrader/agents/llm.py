"""The one wrapper every Claude API call goes through.

It enforces the monthly AI budget, applies the per-agent model and effort from config, asks for
structured (JSON-schema) output with the server-side refusal fallback, and records the cost of
every billed call. Nothing here ever switches a model automatically.

The call uses ``beta.messages.create`` and validates the reply itself rather than using
``beta.messages.parse``: ``parse`` validates every text block eagerly, so a truncated reply or a
refusal that carries text raises out of the SDK before we can record the (billed) usage.
"""

import json
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import anthropic
import pydantic
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from aitrader.calendar import IST, now_ist
from aitrader.config import AIConfig, ModelPrice
from aitrader.store import Store

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class BudgetLevel(StrEnum):
    NORMAL = "normal"
    REDUCED = "reduced"        # >= reduced_at_pct of the cap: effort forced to low
    EXHAUSTED = "exhausted"    # >= 100% of the cap: no AI calls


def budget_level(store: Store, ai_cfg: AIConfig, now: datetime) -> BudgetLevel:
    """Budget state for the IST calendar month containing ``now``."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(IST)
    spent = store.ai_spend_usd(local.year, local.month)
    cap = ai_cfg.monthly_budget_usd
    if spent >= cap:
        return BudgetLevel.EXHAUSTED
    if spent >= cap * ai_cfg.reduced_at_pct / 100:
        return BudgetLevel.REDUCED
    return BudgetLevel.NORMAL


def _tokens(usage: Any) -> tuple[int, int, int, int]:
    """(input, output, cache read, cache write) tokens. A missing or None field counts as 0."""
    def n(field: str) -> int:
        return getattr(usage, field, None) or 0

    return (n("input_tokens"), n("output_tokens"),
            n("cache_read_input_tokens"), n("cache_creation_input_tokens"))


def cost_usd(usage: Any, price: ModelPrice) -> float:
    """US$ cost of one response's token usage."""
    inp, out, cache_read, cache_write = _tokens(usage)
    return (inp * price.input + out * price.output
            + cache_read * price.cache_read + cache_write * price.cache_write) / 1e6


def _final_text(content: Any) -> str:
    """Text of the final model's turn: the text blocks after the last fallback marker, if any."""
    blocks = list(content or [])
    for i in range(len(blocks) - 1, -1, -1):
        if getattr(blocks[i], "type", None) == "fallback":
            blocks = blocks[i + 1:]
            break
    return "".join(b.text for b in blocks if getattr(b, "type", None) == "text")


class LLMClient:
    def __init__(self, store: Store, ai_cfg: AIConfig, client: anthropic.Anthropic | None = None,
                 clock: Callable[[], datetime] = now_ist):
        self.store = store
        self.ai_cfg = ai_cfg
        self.client = client if client is not None else anthropic.Anthropic()
        self.clock = clock

    def run(self, agent: str, system: str, user: str, output_model: type[T],
            max_tokens: int = 16000) -> T | None:
        """Run one agent call and return its parsed output, or ``None`` if there is none.

        ``None`` means: budget exhausted (no call made), the model refused, the reply had no text
        or did not validate against ``output_model`` (for example it was cut off at
        ``max_tokens``), or a transient API failure (rate limit, connection, 5xx; the SDK already
        retried twice). Every call that reached the model is recorded in the ledger, whatever the
        outcome. A ``BadRequestError`` or any other 4xx is a bug and propagates. An agent or
        model missing from config raises ``KeyError``.
        """
        m = self.ai_cfg.agents.get(agent)
        if m is None:
            raise KeyError(f"agent {agent!r} is not configured in ai.agents "
                           f"(configured: {sorted(self.ai_cfg.agents)})")
        price = self.ai_cfg.prices.get(m.model)
        if price is None:
            raise KeyError(f"model {m.model!r} (agent {agent!r}) has no entry in ai.prices "
                           f"(priced: {sorted(self.ai_cfg.prices)})")

        level = budget_level(self.store, self.ai_cfg, self.clock())
        if level is BudgetLevel.EXHAUSTED:
            log.warning("agent=%s skipped: monthly AI budget exhausted", agent)
            return None
        effort = "low" if level is BudgetLevel.REDUCED else m.effort

        try:
            resp = self.client.beta.messages.create(
                model=m.model, max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                # The same schema the SDK's parse() would send for output_format=output_model.
                output_config={"effort": effort,
                               "format": {"type": "json_schema",
                                          "schema": anthropic.transform_schema(output_model)}},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            log.error("agent=%s API call failed: %s", agent, type(e).__name__)
            return None
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                log.error("agent=%s API call failed: %s status=%s", agent, type(e).__name__, e.status_code)
                return None
            raise  # BadRequestError and other 4xx: a bug, not an outage

        # Bill by the model that actually served the reply (a fallback may have replaced ours).
        served = getattr(resp, "model", None) or m.model
        served_price = self.ai_cfg.prices.get(served)
        if served != m.model:
            log.warning("agent=%s configured model %s but %s served the response%s", agent, m.model,
                        served, "" if served_price else f"; it has no ai.prices entry, costing at {m.model}")
        cost = cost_usd(resp.usage, served_price or price)

        parsed, problem = self._parse_output(resp, output_model)

        # Usage is recorded whatever the outcome (refusal, cut off, unparseable): it was billed.
        inp, out, cache_read, cache_write = _tokens(resp.usage)
        with self.store.transaction():
            ts = self.clock()
            self.store.add_ai_usage(served, inp, out, cache_read, cache_write, cost, ts)
            self.store.save_agent_output(agent, self._payload(resp, parsed, problem), cost, ts)

        if problem == "refusal":
            details = resp.stop_details
            log.warning("agent=%s refusal category=%s explanation=%s", agent,
                        getattr(details, "category", None), getattr(details, "explanation", None))
        elif problem == "no_text":
            log.warning("agent=%s reply had no text (stop_reason=%s)", agent, resp.stop_reason)
        elif problem is not None:
            # Only the exception class: its message would echo the model's output.
            log.warning("agent=%s output did not validate: %s (stop_reason=%s)", agent, problem,
                        resp.stop_reason)
        return parsed

    @staticmethod
    def _parse_output(resp: Any, output_model: type[T]) -> tuple[T | None, str | None]:
        """(parsed output, problem). ``problem`` is None on success, else ``"refusal"``,
        ``"no_text"`` or the validation exception's class name. Never raises on bad content."""
        if resp.stop_reason == "refusal":
            return None, "refusal"
        text = _final_text(resp.content)
        if not text.strip():
            return None, "no_text"
        try:
            return output_model.model_validate_json(text), None
        except pydantic.ValidationError as e:
            return None, type(e).__name__

    @staticmethod
    def _payload(resp: Any, parsed: BaseModel | None, problem: str | None) -> str:
        if parsed is not None:
            return parsed.model_dump_json()
        details = resp.stop_details
        return json.dumps({"parsed_output": None, "problem": problem, "stop_reason": resp.stop_reason,
                           "category": getattr(details, "category", None),
                           "explanation": getattr(details, "explanation", None)})


class _Ok(BaseModel):
    ok: bool


def smoke(config_path: Path | str = Path("config.yaml")) -> None:
    """Live check: one tiny structured-output call (about US$0.01). Run via ``python -c``.

    Loads ``.env`` from the repo root (the folder holding ``config.yaml``) so a key that
    lives only there works. The key is never printed or logged. The call goes through
    ``LLMClient.run``, so it is recorded in the real ledger (and counts against the monthly
    budget like any other) and a refusal, cut-off or unparseable reply ends with a clear message
    instead of a traceback.
    """
    logging.basicConfig(level=logging.INFO)
    config_path = Path(config_path).resolve()
    root = config_path.parent
    load_dotenv(root / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(f"ANTHROPIC_API_KEY is not set in the environment or in {root / '.env'}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ai_cfg = AIConfig.model_validate(raw["ai"])
    db_path = Path(raw["db_path"])
    store = Store(db_path if db_path.is_absolute() else root / db_path)
    try:
        llm = LLMClient(store, ai_cfg)
        now = llm.clock()
        before = store.ai_spend_usd(now.year, now.month)
        result = llm.run("news", "You answer with the requested JSON only.",
                         "Set ok to true.", _Ok, max_tokens=1000)
        spent = store.ai_spend_usd(now.year, now.month) - before
        if result is None:
            sys.exit(f"smoke test failed: no usable result (see log above); cost=${spent:.4f}")
        print(f"ok={result.ok} cost=${spent:.4f}")
    finally:
        store.close()
