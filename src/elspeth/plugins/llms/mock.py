"""Mock LLM client for local testing and sample suites."""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar

from elspeth.core.interfaces import LLMClientProtocol


class MockLLMClient(LLMClientProtocol):
    """Deterministic mock LLM for testing pipelines.

    Generates predictable responses based on SHA-256 hash of inputs.
    Useful for local testing and sample suites without API calls.
    """

    # Input: system_prompt, user_prompt, and optional metadata
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["system_prompt", "user_prompt"],
        "properties": {
            "system_prompt": {"type": "string", "description": "System prompt for the LLM"},
            "user_prompt": {"type": "string", "description": "User prompt to process"},
            "metadata": {"type": "object", "description": "Optional metadata context"},
        },
    }

    # Output: content, metrics, and raw request data
    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["content", "metrics"],
        "properties": {
            "content": {"type": "string", "description": "Generated response text"},
            "metrics": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "description": "Derived score (0.4-0.9)"},
                    "comment": {"type": "string"},
                },
            },
            "raw": {"type": "object", "description": "Raw request data for debugging"},
        },
    }

    def __init__(self, *, seed: int | None = None):
        self.seed = seed or 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = metadata or {}
        score = self._derive_score(system_prompt, user_prompt, context)
        return {
            "content": f"[mock] score={score:.2f}\n{user_prompt}",
            "metrics": {
                "score": score,
                "comment": "Mock response generated for demonstration",  # optional helper
            },
            "raw": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "metadata": context,
            },
        }

    def _derive_score(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any]) -> float:
        hasher = hashlib.sha256()
        hasher.update(system_prompt.encode("utf-8"))
        hasher.update(user_prompt.encode("utf-8"))
        if metadata:
            hasher.update(str(sorted(metadata.items())).encode("utf-8"))
        hasher.update(str(self.seed).encode("utf-8"))
        digest = hasher.digest()
        raw = digest[0]
        return 0.4 + (raw / 255.0) * 0.5


# --- Plugin Registration ---
from elspeth.core.registry import registry

MOCK_SCHEMA = {
    "type": "object",
    "properties": {
        "seed": {"type": "integer"},
    },
    "additionalProperties": True,
}

registry.register_llm("mock", MockLLMClient, MOCK_SCHEMA)

__all__ = ["MockLLMClient"]
