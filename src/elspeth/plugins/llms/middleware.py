"""Default LLM middleware implementations."""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from elspeth.core.llm.middleware import LLMMiddleware, LLMRequest
from elspeth.core.llm.registry import register_middleware

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence


_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "include_prompts": {"type": "boolean"},
        "channel": {"type": "string"},
    },
    "additionalProperties": True,
}

_PROMPT_SHIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "denied_terms": {"type": "array", "items": {"type": "string"}},
        "mask": {"type": "string"},
        "on_violation": {"type": "string", "enum": ["abort", "mask", "log"]},
        "channel": {"type": "string"},
    },
    "additionalProperties": True,
}

_HEALTH_SCHEMA = {
    "type": "object",
    "properties": {
        "heartbeat_interval": {"type": "number", "minimum": 0.0},
        "stats_window": {"type": "integer", "minimum": 1},
        "channel": {"type": "string"},
        "include_latency": {"type": "boolean"},
    },
    "additionalProperties": True,
}

_CONTENT_SAFETY_SCHEMA = {
    "type": "object",
    "properties": {
        "endpoint": {"type": "string"},
        "endpoint_env": {"type": "string"},
        "key": {"type": "string"},
        "key_env": {"type": "string"},
        "api_version": {"type": "string"},
        "categories": {"type": "array", "items": {"type": "string"}},
        "severity_threshold": {"type": "integer", "minimum": 0, "maximum": 7},
        "on_violation": {"type": "string", "enum": ["abort", "mask", "log"]},
        "mask": {"type": "string"},
        "channel": {"type": "string"},
        "on_error": {"type": "string", "enum": ["abort", "skip"]},
    },
    "additionalProperties": True,
}

_TRACING_SCHEMA = {
    "type": "object",
    "properties": {
        "sink": {"type": "string", "enum": ["log", "jsonl", "otel"]},
        "path": {"type": "string"},
        "sample_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "include_prompts": {"type": "boolean"},
        "include_response": {"type": "boolean"},
        "max_chars": {"type": "integer", "minimum": 1},
        "redact_fields": {"type": "array", "items": {"type": "string"}},
        "redact_value": {"type": "string"},
        "channel": {"type": "string"},
    },
    "additionalProperties": True,
}

try:  # Optional dependency for OTEL tracing
    from opentelemetry import trace  # type: ignore
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcExporter  # type: ignore
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpExporter  # type: ignore
    from opentelemetry.sdk.resources import Resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
    from opentelemetry.trace import Status, StatusCode  # type: ignore
except Exception:  # pragma: no cover - optional path
    trace = None  # type: ignore
    Status = None  # type: ignore
    StatusCode = None  # type: ignore
    TracerProvider = None  # type: ignore
    Resource = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    OTLPGrpcExporter = None  # type: ignore
    OTLPHttpExporter = None  # type: ignore


class AuditMiddleware(LLMMiddleware):
    name = "audit_logger"

    def __init__(self, *, include_prompts: bool = False, channel: str | None = None):
        self.include_prompts = include_prompts
        self.channel = channel or "dmp.audit"

    def before_request(self, request: LLMRequest) -> LLMRequest:
        payload = {"metadata": request.metadata}
        if self.include_prompts:
            payload.update({"system": request.system_prompt, "user": request.user_prompt})
        logger.info("[%s] LLM request metadata=%s", self.channel, payload)
        return request

    def after_response(self, request: LLMRequest, response: dict[str, Any]) -> dict[str, Any]:
        logger.info("[%s] LLM response metrics=%s", self.channel, response.get("metrics"))
        if self.include_prompts:
            logger.debug("[%s] LLM response content=%s", self.channel, response.get("content"))
        return response


class PromptShieldMiddleware(LLMMiddleware):
    name = "prompt_shield"

    def __init__(
        self,
        *,
        denied_terms: Sequence[str] | None = None,
        mask: str = "[REDACTED]",
        on_violation: str = "abort",
        channel: str | None = None,
    ):
        self.denied_terms = [term.lower() for term in denied_terms or []]
        self.mask = mask
        mode = (on_violation or "abort").lower()
        if mode not in {"abort", "mask", "log"}:
            mode = "abort"
        self.mode = mode
        self.channel = channel or "dmp.prompt_shield"

    def before_request(self, request: LLMRequest) -> LLMRequest:
        lowered = request.user_prompt.lower()
        for term in self.denied_terms:
            if term and term in lowered:
                logger.warning("[%s] Prompt contains blocked term '%s'", self.channel, term)
                if self.mode == "abort":
                    raise ValueError(f"Prompt contains blocked term '{term}'")
                if self.mode == "mask":
                    masked = request.user_prompt.replace(term, self.mask)
                    return request.clone(user_prompt=masked)
                break
        return request


class HealthMonitorMiddleware(LLMMiddleware):
    """Emit heartbeat logs summarising middleware activity."""

    name = "health_monitor"

    def __init__(
        self,
        *,
        heartbeat_interval: float = 60.0,
        stats_window: int = 50,
        channel: str | None = None,
        include_latency: bool = True,
    ) -> None:
        if heartbeat_interval < 0:
            raise ValueError("heartbeat_interval must be non-negative")
        self.interval = float(heartbeat_interval)
        self.window = max(int(stats_window), 1)
        self.channel = channel or "dmp.health"
        self.include_latency = include_latency
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=self.window)
        self._inflight: dict[int, float] = {}
        self._total_requests = 0
        self._total_failures = 0
        self._last_heartbeat = time.monotonic()

    def before_request(self, request: LLMRequest) -> LLMRequest:
        start = time.monotonic()
        with self._lock:
            self._inflight[id(request)] = start
        return request

    def after_response(self, request: LLMRequest, response: dict[str, Any]) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            start = self._inflight.pop(id(request), None)
            self._total_requests += 1
            if isinstance(response, dict) and response.get("error"):
                self._total_failures += 1
            if start is not None and self.include_latency:
                self._latencies.append(now - start)
            if self.interval == 0 or now - self._last_heartbeat >= self.interval:
                self._emit(now)
        return response

    def _emit(self, now: float) -> None:
        data: dict[str, Any] = {
            "requests": self._total_requests,
            "failures": self._total_failures,
        }
        if self._total_requests:
            data["failure_rate"] = self._total_failures / self._total_requests
        if self.include_latency and self._latencies:
            latencies = list(self._latencies)
            count = len(latencies)
            total = sum(latencies)
            data.update(
                {
                    "latency_count": count,
                    "latency_avg": total / count,
                    "latency_min": min(latencies),
                    "latency_max": max(latencies),
                }
            )
        logger.info("[%s] health heartbeat %s", self.channel, data)
        self._last_heartbeat = now


class AzureContentSafetyMiddleware(LLMMiddleware):
    """Use Azure Content Safety service to screen prompts before submission."""

    name = "azure_content_safety"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        endpoint_env: str | None = None,
        key: str | None = None,
        key_env: str | None = None,
        api_version: str | None = None,
        categories: Sequence[str] | None = None,
        severity_threshold: int = 4,
        on_violation: str = "abort",
        mask: str = "[CONTENT BLOCKED]",
        channel: str | None = None,
        on_error: str = "abort",
    ) -> None:
        endpoint_value = endpoint or (os.environ.get(endpoint_env) if endpoint_env else None)
        if not endpoint_value:
            raise ValueError("Azure Content Safety requires an endpoint or endpoint_env")
        self.endpoint = endpoint_value.rstrip("/")
        key_value = key or (os.environ.get(key_env) if key_env else None)
        if not key_value:
            raise ValueError("Azure Content Safety requires an API key or key_env")
        self.key = key_value
        self.api_version = api_version or "2023-10-01"
        self.categories = list(categories or ["Hate", "Violence", "SelfHarm", "Sexual"])
        self.threshold = max(0, min(int(severity_threshold), 7))
        mode = (on_violation or "abort").lower()
        if mode not in {"abort", "mask", "log"}:
            mode = "abort"
        self.mode = mode
        self.mask = mask
        self.channel = channel or "dmp.azure_content_safety"
        handler = (on_error or "abort").lower()
        if handler not in {"abort", "skip"}:
            handler = "abort"
        self.on_error = handler

    def before_request(self, request: LLMRequest) -> LLMRequest:
        try:
            result = self._analyze_text(request.user_prompt)
        except Exception as exc:  # pragma: no cover - network failure path
            if self.on_error == "skip":
                logger.warning("[%s] Content Safety call failed; skipping (%s)", self.channel, exc)
                return request
            raise

        if result.get("flagged"):
            logger.warning("[%s] Prompt flagged by Azure Content Safety: %s", self.channel, result)
            if self.mode == "abort":
                raise ValueError("Prompt blocked by Azure Content Safety")
            if self.mode == "mask":
                return request.clone(user_prompt=self.mask)
        return request

    def _analyze_text(self, text: str) -> dict[str, Any]:
        url = f"{self.endpoint}/contentsafety/text:analyze?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.key,
        }
        payload = {
            "text": text,
            "categories": self.categories,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        flagged = False
        max_severity = 0
        for item in data.get("results", data.get("categories", [])):
            severity = int(item.get("severity", 0))
            max_severity = max(max_severity, severity)
            if severity >= self.threshold:
                flagged = True
        return {"flagged": flagged, "max_severity": max_severity, "raw": data}


class TracingMiddleware(LLMMiddleware):
    """Lightweight trace emitter for any LLM client (OpenAI-compatible or Azure)."""

    name = "tracing"

    def __init__(
        self,
        *,
        sink: str = "log",
        path: str | None = None,
        sample_rate: float = 1.0,
        include_prompts: bool = False,
        include_response: bool = False,
        max_chars: int = 8000,
        redact_fields: Sequence[str] | None = None,
        redact_value: str = "[REDACTED]",
        channel: str | None = None,
    ):
        if sink not in {"log", "jsonl", "otel"}:
            raise ValueError("Tracing sink must be 'log', 'jsonl', or 'otel'")
        if sample_rate < 0 or sample_rate > 1:
            raise ValueError("sample_rate must be between 0 and 1")
        self.sink = sink
        self.path = Path(path) if path else Path("llm_traces.jsonl")
        self.sample_rate = float(sample_rate)
        self.include_prompts = include_prompts
        self.include_response = include_response
        self.max_chars = max(int(max_chars), 1)
        self.redact_fields = {field.lower() for field in (redact_fields or [])}
        self.redact_value = redact_value
        self.channel = channel or "dmp.trace"
        self._inflight: dict[int, float] = {}
        self._spans: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._tracer = self._resolve_tracer() if sink == "otel" else None
        if sink == "otel":
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if not endpoint:
                raise RuntimeError("TracingMiddleware sink=otel requires OTEL_EXPORTER_OTLP_ENDPOINT to be set to a collector URL")
            if self._tracer is None:
                raise RuntimeError("TracingMiddleware sink=otel requires opentelemetry SDK to be installed and configured")
            # Minimal exporter setup if none is present
            provider = trace.get_tracer_provider()
            needs_provider = True
            if TracerProvider and isinstance(provider, TracerProvider):
                # Check if any span processors are attached
                needs_provider = not bool(getattr(provider, "_active_span_processor", None))
            if needs_provider:
                if not (BatchSpanProcessor and Resource and (OTLPGrpcExporter or OTLPHttpExporter)):
                    raise RuntimeError("TracingMiddleware sink=otel could not configure an OTLP exporter (missing SDK components)")
                proto = (os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL") or "grpc").lower()
                use_http = proto.startswith("http")
                exporter_cls = OTLPHttpExporter if use_http else OTLPGrpcExporter  # type: ignore[assignment]
                service_name = os.getenv("OTEL_SERVICE_NAME", "elspeth")
                resource = Resource.create({"service.name": service_name})  # type: ignore[arg-type]
                new_provider = TracerProvider(resource=resource)  # type: ignore[call-arg]
                processor = BatchSpanProcessor(exporter_cls())  # type: ignore[arg-type]
                new_provider.add_span_processor(processor)
                trace.set_tracer_provider(new_provider)
                self._tracer = trace.get_tracer("elspeth.llm.tracing")
            if endpoint.startswith(("http://", "https://")):
                ping_url = endpoint.rstrip("/") + "/v1/traces"
                try:
                    # We only care about reachability; any HTTP status means the listener responded.
                    requests.get(ping_url, timeout=2)
                except Exception as exc:  # pragma: no cover - network refusal path
                    raise RuntimeError(f"TracingMiddleware sink=otel cannot reach collector at {ping_url}") from exc

    def before_request(self, request: LLMRequest) -> LLMRequest:
        if not self._should_sample():
            return request
        stamp = time.time()
        with self._lock:
            self._inflight[id(request)] = stamp
        payload = self._base_payload("start", stamp, request.metadata)
        if self.include_prompts:
            payload["system_prompt"] = self._truncate(request.system_prompt)
            payload["user_prompt"] = self._truncate(request.user_prompt)
        if self.sink == "otel" and self._tracer:
            span = self._tracer.start_span("llm.call")
            with self._lock:
                self._spans[id(request)] = span
            self._apply_span_attributes(span, payload, include_response=False)
        else:
            self._emit(payload)
        return request

    def after_response(self, request: LLMRequest, response: dict[str, Any]) -> dict[str, Any]:
        start = None
        span = None
        with self._lock:
            start = self._inflight.pop(id(request), None)
            span = self._spans.pop(id(request), None)
        stamp = time.time()
        payload = self._base_payload("end", stamp, request.metadata)
        if start is not None:
            payload["latency_ms"] = max((stamp - start) * 1000, 0.0)
        if isinstance(response, dict):
            metrics = response.get("metrics") or {}
            payload["status"] = "error" if response.get("error") else "ok"
            payload["tokens_prompt"] = metrics.get("prompt_tokens")
            payload["tokens_completion"] = metrics.get("completion_tokens")
            payload["cost"] = metrics.get("cost")
            retry = response.get("retry")
            if isinstance(retry, dict):
                payload["attempts_used"] = retry.get("attempts")
                payload["attempts_max"] = retry.get("max_attempts")
            if self.include_response:
                payload["response_content"] = self._truncate(str(response.get("content")))
        if self.sink == "otel" and span is not None:
            self._apply_span_attributes(span, payload, include_response=self.include_response)
            if response and response.get("error") and Status and StatusCode:
                span.set_status(Status(StatusCode.ERROR, description=str(response.get("error"))))
            span.end()
        else:
            self._emit(payload)
        return response

    def _base_payload(self, event: str, timestamp: float, metadata: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": f"llm_{event}",
            "timestamp": timestamp,
        }
        for key, value in (metadata or {}).items():
            key_lower = str(key).lower()
            if key_lower in self.redact_fields:
                payload[key] = self.redact_value
            else:
                payload[key] = value
        return payload

    def _emit(self, payload: dict[str, Any]) -> None:
        try:
            if self.sink == "log":
                logger.info("[%s] %s", self.channel, payload)
                return
            line = payload.copy()
            if isinstance(line.get("timestamp"), float):
                line["timestamp"] = round(line["timestamp"], 6)
            text = f"{line}\n"
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(text)
        except Exception:  # pragma: no cover - tracing must not break the pipeline
            logger.debug("TracingMiddleware emit failed", exc_info=True)

    def _truncate(self, text: str | None) -> str | None:
        if text is None:
            return None
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars] + "...[truncated]"

    def _should_sample(self) -> bool:
        if self.sample_rate >= 1.0:
            return True
        return random.random() <= self.sample_rate

    def _resolve_tracer(self):
        if trace is None:  # type: ignore[truthy-function]
            return None
        try:
            return trace.get_tracer("elspeth.llm.tracing")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - defensive
            return None

    def _apply_span_attributes(self, span: Any, payload: dict[str, Any], *, include_response: bool) -> None:
        if span is None:
            return
        for key, value in payload.items():
            if key in {"event", "timestamp"}:
                continue
            if value is None:
                continue
            attr_key = f"llm.{key}"
            try:
                span.set_attribute(attr_key, value)
            except Exception:  # pragma: no cover - best effort
                logger.debug("Failed to set OTEL attribute %s", attr_key, exc_info=True)
        if include_response and payload.get("response_content"):
            try:
                span.add_event("llm.response", {"response": payload.get("response_content")})
            except Exception:  # pragma: no cover - best effort
                logger.debug("Failed to add OTEL response event", exc_info=True)


register_middleware(
    "audit_logger",
    lambda options: AuditMiddleware(
        include_prompts=bool(options.get("include_prompts", False)),
        channel=options.get("channel"),
    ),
    schema=_AUDIT_SCHEMA,
)

register_middleware(
    "prompt_shield",
    lambda options: PromptShieldMiddleware(
        denied_terms=options.get("denied_terms", []),
        mask=options.get("mask", "[REDACTED]"),
        on_violation=options.get("on_violation", "abort"),
        channel=options.get("channel"),
    ),
    schema=_PROMPT_SHIELD_SCHEMA,
)

register_middleware(
    "health_monitor",
    lambda options: HealthMonitorMiddleware(
        heartbeat_interval=float(options.get("heartbeat_interval", 60.0)),
        stats_window=int(options.get("stats_window", 50)),
        channel=options.get("channel"),
        include_latency=bool(options.get("include_latency", True)),
    ),
    schema=_HEALTH_SCHEMA,
)

register_middleware(
    "azure_content_safety",
    lambda options: AzureContentSafetyMiddleware(
        endpoint=options.get("endpoint"),
        endpoint_env=options.get("endpoint_env"),
        key=options.get("key"),
        key_env=options.get("key_env"),
        api_version=options.get("api_version"),
        categories=options.get("categories"),
        severity_threshold=int(options.get("severity_threshold", 4)),
        on_violation=options.get("on_violation", "abort"),
        mask=options.get("mask", "[CONTENT BLOCKED]"),
        channel=options.get("channel"),
        on_error=options.get("on_error", "abort"),
    ),
    schema=_CONTENT_SAFETY_SCHEMA,
)

register_middleware(
    "tracing",
    lambda options: TracingMiddleware(
        sink=options.get("sink", "log"),
        path=options.get("path"),
        sample_rate=float(options.get("sample_rate", 1.0)),
        include_prompts=bool(options.get("include_prompts", False)),
        include_response=bool(options.get("include_response", False)),
        max_chars=int(options.get("max_chars", 8000)),
        redact_fields=options.get("redact_fields", []),
        redact_value=options.get("redact_value", "[REDACTED]"),
        channel=options.get("channel"),
    ),
    schema=_TRACING_SCHEMA,
)


__all__ = [
    "AuditMiddleware",
    "AzureContentSafetyMiddleware",
    "HealthMonitorMiddleware",
    "PromptShieldMiddleware",
    "TracingMiddleware",
]
