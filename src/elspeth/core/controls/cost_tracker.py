"""Cost tracker protocols and implementations."""

from __future__ import annotations

from typing import Any


class CostTracker:
    """Tracks LLM costs per request and aggregates totals."""

    def record(self, response: dict[str, Any], metadata: dict[str, object] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def summary(self) -> dict[str, Any]:
        raise NotImplementedError


class NoopCostTracker(CostTracker):  # pragma: no cover - trivial
    def record(self, response: dict[str, Any], metadata: dict[str, object] | None = None) -> dict[str, Any]:
        return {}

    def summary(self) -> dict[str, Any]:
        return {}


class FixedPriceCostTracker(CostTracker):
    """Cost tracker using fixed per-token pricing."""

    def __init__(self, prompt_token_price: float = 0.0, completion_token_price: float = 0.0):
        self.prompt_token_price = prompt_token_price
        self.completion_token_price = completion_token_price
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

    def record(self, response: dict[str, Any], metadata: dict[str, object] | None = None) -> dict[str, Any]:
        usage = self._extract_usage(response.get("raw"))
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        cost = prompt_tokens * self.prompt_token_price + completion_tokens * self.completion_token_price

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += cost

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_cost": self.total_cost,
        }

    @staticmethod
    def _extract_usage(raw: Any) -> dict[str, int]:
        if raw is None:
            return {}
        usage = None
        if hasattr(raw, "usage"):
            usage = getattr(raw.usage, "to_dict", lambda: raw.usage)()
        elif isinstance(raw, dict):
            usage = raw.get("usage")
        if usage is None:
            return {}
        if hasattr(usage, "to_dict"):
            usage = usage.to_dict()
        if not isinstance(usage, dict):
            return {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        }


__all__ = [
    "CostTracker",
    "FixedPriceCostTracker",
    "NoopCostTracker",
]
