"""Cost tracking: token estimation and cost calculation.

Supports both OpenAI-style (tiktoken) and Anthropic/Ark-style (character-based
heuristic) token counting.  Prices are stored in a lookup table that can be
overridden via environment variables.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from agent.observability import LLM_TOKENS_TOTAL, LLM_COST_USD_TOTAL, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens)
# ---------------------------------------------------------------------------

_DEFAULT_PRICES: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"prompt": 5.0, "completion": 15.0},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
    "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"prompt": 3.0, "completion": 15.0},
    "claude-3-opus-20240229": {"prompt": 15.0, "completion": 75.0},
    "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
    # Ark / Volcengine (approximate, update as needed)
    "ark-code-latest": {"prompt": 2.0, "completion": 8.0},
    "kimi-k3": {"prompt": 2.0, "completion": 8.0},
    # Gemini
    "gemini-1.5-pro": {"prompt": 3.5, "completion": 10.5},
    "gemini-1.5-flash": {"prompt": 0.35, "completion": 1.05},
}


def _get_prices() -> dict[str, dict[str, float]]:
    """Return pricing table, allowing override via ``LLM_PRICES_JSON``."""
    import json

    override = os.environ.get("LLM_PRICES_JSON")
    if override:
        try:
            return json.loads(override)
        except Exception as exc:
            logger.warning("invalid_llm_prices_json", error=str(exc))
    return _DEFAULT_PRICES


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str, model: str) -> int:
    """Estimate token count for *text*.

    Uses tiktoken when available and the model is an OpenAI model; otherwise
    falls back to a conservative character heuristic (≈ 4 chars / token for
    CJK, ≈ 4 chars / token for English).
    """
    if not text:
        return 0

    # Try tiktoken for OpenAI-style models
    if any(m in model.lower() for m in ("gpt-", "text-")):
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass

    # Fallback heuristic: ~4 characters per token on average
    return max(1, len(text) // 4)


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Return estimated cost in USD.

    Cost = (prompt_tokens / 1_000_000 * prompt_price) + (completion_tokens / 1_000_000 * completion_price)
    """
    prices = _get_prices()
    model_key = model.lower()

    # Exact match first, then prefix match
    price = prices.get(model_key)
    if price is None:
        for k, v in prices.items():
            if model_key.startswith(k.lower()):
                price = v
                break

    if price is None:
        # Unknown model — use a safe default so metrics still work
        logger.debug("unknown_model_for_costing", model=model)
        price = {"prompt": 5.0, "completion": 15.0}

    prompt_cost = prompt_tokens / 1_000_000 * price["prompt"]
    completion_cost = completion_tokens / 1_000_000 * price["completion"]
    return round(prompt_cost + completion_cost, 8)


# ---------------------------------------------------------------------------
# Helper to record metrics after an LLM call
# ---------------------------------------------------------------------------


def record_llm_usage(
    model: str,
    stage: str,
    prompt_text: str,
    completion_text: str,
) -> Tuple[int, int, float]:
    """Estimate tokens, calculate cost, and update Prometheus counters.

    Returns:
        (prompt_tokens, completion_tokens, cost_usd)
    """
    prompt_tokens = estimate_tokens(prompt_text, model)
    completion_tokens = estimate_tokens(completion_text, model)
    cost = calculate_cost(prompt_tokens, completion_tokens, model)

    LLM_TOKENS_TOTAL.labels(model=model, stage=stage, token_type="prompt").inc(prompt_tokens)
    LLM_TOKENS_TOTAL.labels(model=model, stage=stage, token_type="completion").inc(completion_tokens)
    LLM_COST_USD_TOTAL.labels(model=model, stage=stage).inc(cost)

    logger.info(
        "llm_usage_recorded",
        model=model,
        stage=stage,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
    )
    return prompt_tokens, completion_tokens, cost
