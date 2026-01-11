"""OpenRouter client wrapper implementing the LLM protocol."""

from __future__ import annotations

import os
from typing import Any

from elspeth.core.interfaces import LLMClientProtocol


class OpenRouterClient(LLMClientProtocol):
    """OpenRouter LLM client compatible with OpenAI API.

    OpenRouter provides access to multiple LLM providers through a unified API.
    See https://openrouter.ai for available models and pricing.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        config: dict[str, Any],
        client: Any | None = None,
    ):
        self.config = config
        self.temperature = config.get("temperature")
        self.max_tokens = config.get("max_tokens")
        self.model = self._resolve_model(model)
        self._client = client or self._create_client()

    def _create_client(self):
        api_key = self._resolve_required("api_key")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency ensured in runtime
            raise RuntimeError("openai package is required for OpenRouterClient") from exc

        # OpenRouter uses OpenAI-compatible API
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def _resolve_model(self, model: str | None) -> str:
        """Resolve the model name from config or environment."""
        if model:
            return model
        if self.config.get("model"):
            return str(self.config["model"])
        env_key = self.config.get("model_env")
        if env_key:
            value = os.getenv(env_key)
            if value:
                return value
        value = os.getenv("OPENROUTER_MODEL")
        if value:
            return value
        # Default to a reasonable model
        return "openai/gpt-4o-mini"

    def _resolve_required(self, key: str) -> str:
        value = self._resolve_optional(key)
        if not value:
            raise ValueError(f"OpenRouterClient missing required config value '{key}'")
        return value

    def _resolve_optional(self, key: str) -> str | None:
        if self.config.get(key):
            return str(self.config[key])
        env_key = self.config.get(f"{key}_env")
        if env_key:
            return os.getenv(env_key)
        return None

    @property
    def client(self):  # type: ignore[return-any]
        return self._client

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a completion using OpenRouter."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        # Add OpenRouter-specific headers if provided
        extra_headers = {}
        if self.config.get("site_url"):
            extra_headers["HTTP-Referer"] = self.config["site_url"]
        if self.config.get("app_name"):
            extra_headers["X-Title"] = self.config["app_name"]

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        response = self.client.chat.completions.create(**kwargs)
        content = None
        try:
            content = response.choices[0].message.content
        except Exception:  # pragma: no cover - defensive fallback
            content = None

        return {
            "content": content,
            "raw": response,
            "metadata": metadata or {},
        }


# --- Plugin Registration ---
from elspeth.core.registry import registry

OPENROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "config": {"type": "object"},
        "model": {"type": "string"},
        "client": {},
    },
    "required": ["config"],
    "additionalProperties": True,
}

registry.register_llm("openrouter", OpenRouterClient, OPENROUTER_SCHEMA)
