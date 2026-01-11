"""Middleware primitives for LLM interactions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol


@dataclass
class LLMRequest:
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]

    def clone(
        self,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMRequest:
        return replace(
            self,
            system_prompt=system_prompt if system_prompt is not None else self.system_prompt,
            user_prompt=user_prompt if user_prompt is not None else self.user_prompt,
            metadata=metadata if metadata is not None else dict(self.metadata),
        )


class LLMMiddleware(Protocol):
    """Intercepts requests/responses around LLM calls."""

    name: str

    def before_request(self, request: LLMRequest) -> LLMRequest:
        return request

    def after_response(self, request: LLMRequest, response: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - optional override
        return response


__all__ = ["LLMMiddleware", "LLMRequest"]
