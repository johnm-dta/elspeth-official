"""Azure OpenAI client wrapper implementing the LLM protocol."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from elspeth.core.interfaces import LLMClientProtocol

logger = logging.getLogger(__name__)


class AzureOpenAIClient(LLMClientProtocol):
    def __init__(
        self,
        *,
        deployment: str | None = None,
        config: dict[str, Any],
        client: Any | None = None,
    ):
        self.config = config
        self.temperature = config.get("temperature")
        self.max_tokens = config.get("max_tokens")
        self.response_format = config.get("response_format")  # "json_object" or None
        self.deployment = self._resolve_deployment(deployment)
        self._client = client or self._create_client()

    def _create_client(self):
        api_key = self._resolve_required("api_key")
        api_version = self._resolve_required("api_version")
        azure_endpoint = self._resolve_required("azure_endpoint")

        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - dependency ensured in runtime
            raise RuntimeError("openai package is required for AzureOpenAIClient") from exc

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - dependency ensured in runtime
            raise RuntimeError("httpx package is required for AzureOpenAIClient") from exc

        # Use custom httpx client with high connection pool limits for parallel requests
        # Without this, the default client has low limits that cause request queueing
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=300,
                keepalive_expiry=60,
            ),
            timeout=httpx.Timeout(30.0),
        )

        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            http_client=http_client,
        )

    def _resolve_deployment(self, deployment: str | None) -> str:
        if deployment:
            return deployment
        if self.config.get("deployment"):
            return self.config["deployment"]
        env_key = self.config.get("deployment_env")
        if env_key:
            value = os.getenv(env_key)
            if value:
                return value
        value = os.getenv("DMP_AZURE_OPENAI_DEPLOYMENT")
        if value:
            return value
        raise ValueError("AzureOpenAIClient missing deployment configuration")

    def _resolve_required(self, key: str) -> str:
        value = self._resolve_optional(key)
        if not value:
            raise ValueError(f"AzureOpenAIClient missing required config value '{key}'")
        return value

    def _resolve_optional(self, key: str) -> str | None:
        if self.config.get(key):
            return self.config[key]
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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict[str, Any] = {"model": self.deployment, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.response_format:
            kwargs["response_format"] = {"type": self.response_format}

        response = self.client.chat.completions.create(**kwargs)
        content = None
        try:
            content = response.choices[0].message.content
        except Exception:  # pragma: no cover - defensive fallback
            content = None

        # Convert response to dict for JSON serializability
        raw_dict = response.model_dump() if hasattr(response, "model_dump") else str(response)

        result = {
            "content": content,
            "raw": raw_dict,
            "metadata": metadata or {},
        }

        # Parse JSON response if response_format is json_object
        if self.response_format == "json_object" and content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    # Flatten parsed JSON fields into result
                    for key, value in parsed.items():
                        result[key] = value
                    logger.debug("Parsed JSON response with keys: %s", list(parsed.keys()))
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse JSON response: %s", e)

        return result


# --- Plugin Registration ---
from elspeth.core.registry import registry

AZURE_OPENAI_SCHEMA = {
    "type": "object",
    "properties": {
        "config": {"type": "object"},
        "deployment": {"type": "string"},
        "client": {},
    },
    "required": ["config"],
    "additionalProperties": True,
}

registry.register_llm("azure_openai", AzureOpenAIClient, AZURE_OPENAI_SCHEMA)
