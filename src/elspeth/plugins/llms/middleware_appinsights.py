"""Azure Application Insights middleware for batch telemetry."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from elspeth.core.llm.middleware import LLMMiddleware, LLMRequest
from elspeth.core.llm.registry import register_middleware

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)

_APPINSIGHTS_SCHEMA = {
    "type": "object",
    "properties": {
        "connection_string": {"type": "string"},
        "connection_string_env": {"type": "string"},
        "instrumentation_key": {"type": "string"},
        "instrumentation_key_env": {"type": "string"},
        "role_name": {"type": "string"},
        "role_instance": {"type": "string"},
        "enable_batch_events": {"type": "boolean"},
        "enable_llm_events": {"type": "boolean"},
        "enable_dependency_tracking": {"type": "boolean"},
        "on_error": {"type": "string", "enum": ["abort", "skip"]},
    },
    "additionalProperties": True,
}


def _get_telemetry_client(
    connection_string: str | None = None,
    instrumentation_key: str | None = None,
    role_name: str | None = None,
    role_instance: str | None = None,
) -> Any | None:
    """Initialize Application Insights TelemetryClient."""
    try:
        from applicationinsights import TelemetryClient
        from applicationinsights.channel import AsynchronousQueue, AsynchronousSender, TelemetryChannel
    except ImportError:
        logger.warning(
            "applicationinsights package not installed. "
            "Install with: pip install applicationinsights"
        )
        return None

    # Connection string takes precedence
    if connection_string:
        # Extract instrumentation key from connection string
        parts = dict(part.split("=", 1) for part in connection_string.split(";") if "=" in part)
        ikey = parts.get("InstrumentationKey")
        if not ikey:
            logger.error("Invalid connection string - missing InstrumentationKey")
            return None
    elif instrumentation_key:
        ikey = instrumentation_key
    else:
        logger.error("No connection_string or instrumentation_key provided")
        return None

    # Create async channel for non-blocking sends
    sender = AsynchronousSender()
    queue = AsynchronousQueue(sender)
    channel = TelemetryChannel(None, queue)

    client = TelemetryClient(ikey, channel)

    # Set cloud role for filtering in App Insights
    client.context.cloud.role = role_name or "elspeth"
    client.context.cloud.role_instance = role_instance or os.environ.get("HOSTNAME", "unknown")

    return client


class AppInsightsMiddleware(LLMMiddleware):
    """Send batch and LLM telemetry to Azure Application Insights.

    Tracks:
    - Batch lifecycle (start, complete, error)
    - Row processing metrics
    - LLM call dependencies (optional)
    - Errors and exceptions
    """

    name = "app_insights"

    def __init__(
        self,
        *,
        connection_string: str | None = None,
        connection_string_env: str | None = None,
        instrumentation_key: str | None = None,
        instrumentation_key_env: str | None = None,
        role_name: str | None = None,
        role_instance: str | None = None,
        enable_batch_events: bool = True,
        enable_llm_events: bool = False,
        enable_dependency_tracking: bool = True,
        on_error: str = "skip",
    ) -> None:
        if on_error not in {"abort", "skip"}:
            raise ValueError("on_error must be 'abort' or 'skip'")

        self.on_error = on_error
        self.enable_batch_events = enable_batch_events
        self.enable_llm_events = enable_llm_events
        self.enable_dependency_tracking = enable_dependency_tracking

        # Resolve connection string
        conn_str = connection_string or (
            os.environ.get(connection_string_env) if connection_string_env else None
        )
        ikey = instrumentation_key or (
            os.environ.get(instrumentation_key_env) if instrumentation_key_env else None
        )

        # Fall back to standard env var
        if not conn_str and not ikey:
            conn_str = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if not conn_str and not ikey:
            ikey = os.environ.get("APPINSIGHTS_INSTRUMENTATIONKEY")

        self._client = _get_telemetry_client(
            connection_string=conn_str,
            instrumentation_key=ikey,
            role_name=role_name,
            role_instance=role_instance,
        )

        if not self._client:
            msg = "Failed to initialize Application Insights client"
            if on_error == "abort":
                raise RuntimeError(msg)
            logger.warning("%s - telemetry will be disabled", msg)
        else:
            logger.info("AppInsightsMiddleware initialized (role=%s)", role_name or "elspeth")

        # Tracking state
        self._lock = threading.Lock()
        self._batch_start: float | None = None
        self._experiment_starts: dict[str, float] = {}
        self._llm_calls: dict[str, float] = {}
        self._sequence = 0

        # Batch metrics
        self._metrics = {
            "total_rows": 0,
            "successful_rows": 0,
            "failed_rows": 0,
            "total_llm_calls": 0,
            "total_llm_duration_ms": 0.0,
        }

    def _next_sequence(self) -> str:
        with self._lock:
            self._sequence += 1
            return f"ai-{self._sequence}"

    # ---- LLM Call Tracking ----

    def before_request(self, request: LLMRequest) -> LLMRequest:
        if not self._client:
            return request

        seq = self._next_sequence()
        with self._lock:
            self._llm_calls[seq] = time.time()
            self._metrics["total_llm_calls"] += 1

        if self.enable_llm_events:
            self._client.track_event(
                "LLMRequest",
                properties={
                    "sequence": seq,
                    "model": request.metadata.get("model", "unknown"),
                    "query_name": request.metadata.get("query_name", "unknown"),
                },
            )

        updated_metadata = dict(request.metadata)
        updated_metadata["appinsights_sequence"] = seq
        return request.clone(metadata=updated_metadata)

    def after_response(self, request: LLMRequest, response: dict[str, Any]) -> dict[str, Any]:
        if not self._client:
            return response

        seq = request.metadata.get("appinsights_sequence")
        duration_ms = 0.0

        if seq:
            with self._lock:
                start = self._llm_calls.pop(seq, None)
                if start:
                    duration_ms = (time.time() - start) * 1000
                    self._metrics["total_llm_duration_ms"] += duration_ms

        # Track as dependency (shows in App Insights Application Map)
        if self.enable_dependency_tracking:
            success = "error" not in response
            self._client.track_dependency(
                name=request.metadata.get("query_name", "LLM Call"),
                data=request.metadata.get("model", ""),
                type="LLM",
                target=request.metadata.get("deployment", "azure-openai"),
                duration=duration_ms,
                success=success,
                dependency_type_name="Azure OpenAI",
            )

        return response

    def on_error(self, request: LLMRequest, error: Exception) -> None:
        if not self._client:
            return

        self._client.track_exception()

        with self._lock:
            self._metrics["failed_rows"] += 1

    # ---- Batch Lifecycle ----

    def on_suite_loaded(
        self,
        experiments: Iterable[Mapping[str, Any]],
        preflight: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._client or not self.enable_batch_events:
            return

        experiments = list(experiments)
        self._batch_start = time.time()

        self._client.track_event(
            "BatchStarted",
            properties={
                "experiment_count": str(len(experiments)),
                "experiments": ",".join(e.get("name", "unknown") for e in experiments[:10]),
            },
        )
        self._client.flush()

    def on_experiment_start(self, name: str, metadata: Mapping[str, Any]) -> None:
        if not self._client or not self.enable_batch_events:
            return

        with self._lock:
            self._experiment_starts[name] = time.time()

        row_count = metadata.get("row_count", 0)

        self._client.track_event(
            "ExperimentStarted",
            properties={
                "experiment": name,
                "row_count": str(row_count),
            },
        )

    def on_experiment_complete(
        self,
        name: str,
        payload: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._client or not self.enable_batch_events:
            return

        with self._lock:
            start = self._experiment_starts.pop(name, None)

        duration_ms = (time.time() - start) * 1000 if start else 0

        results = payload.get("results", [])
        failures = payload.get("failures", [])
        row_count = len(results) if isinstance(results, list) else 0
        failure_count = len(failures) if isinstance(failures, list) else 0

        with self._lock:
            self._metrics["total_rows"] += row_count
            self._metrics["successful_rows"] += row_count - failure_count
            self._metrics["failed_rows"] += failure_count

        # Track as request (shows in App Insights Performance)
        self._client.track_request(
            name=f"Experiment:{name}",
            url=f"/experiment/{name}",
            success=failure_count == 0,
            duration=duration_ms,
            response_code="200" if failure_count == 0 else "500",
            properties={
                "rows_processed": str(row_count),
                "failures": str(failure_count),
            },
        )

        # Track metrics
        self._client.track_metric("RowsProcessed", row_count, properties={"experiment": name})
        self._client.track_metric("RowFailures", failure_count, properties={"experiment": name})
        self._client.track_metric("ExperimentDurationMs", duration_ms, properties={"experiment": name})

        if row_count > 0:
            throughput = row_count / (duration_ms / 1000) if duration_ms > 0 else 0
            self._client.track_metric("RowsPerSecond", throughput, properties={"experiment": name})

    def on_suite_complete(self) -> None:
        if not self._client or not self.enable_batch_events:
            return

        duration_ms = (time.time() - self._batch_start) * 1000 if self._batch_start else 0

        with self._lock:
            metrics = dict(self._metrics)

        self._client.track_event(
            "BatchCompleted",
            properties={
                "total_rows": str(metrics["total_rows"]),
                "successful_rows": str(metrics["successful_rows"]),
                "failed_rows": str(metrics["failed_rows"]),
                "total_llm_calls": str(metrics["total_llm_calls"]),
                "duration_ms": str(int(duration_ms)),
            },
            measurements={
                "total_rows": metrics["total_rows"],
                "successful_rows": metrics["successful_rows"],
                "failed_rows": metrics["failed_rows"],
                "total_llm_calls": metrics["total_llm_calls"],
                "duration_ms": duration_ms,
                "avg_llm_duration_ms": (
                    metrics["total_llm_duration_ms"] / metrics["total_llm_calls"]
                    if metrics["total_llm_calls"] > 0 else 0
                ),
            },
        )

        # Final metrics
        self._client.track_metric("BatchDurationMs", duration_ms)
        self._client.track_metric("BatchTotalRows", metrics["total_rows"])
        self._client.track_metric("BatchSuccessRate",
            metrics["successful_rows"] / metrics["total_rows"] * 100
            if metrics["total_rows"] > 0 else 100
        )

        # Flush to ensure all telemetry is sent
        self._client.flush()
        logger.info(
            "AppInsights batch telemetry flushed: %d rows, %d failures, %.1fs",
            metrics["total_rows"],
            metrics["failed_rows"],
            duration_ms / 1000,
        )

    def on_retry_exhausted(self, request: LLMRequest, metadata: Mapping[str, Any], error: Exception) -> None:
        if not self._client:
            return

        self._client.track_exception()
        self._client.track_event(
            "LLMRetryExhausted",
            properties={
                "query_name": request.metadata.get("query_name", "unknown"),
                "attempts": str(metadata.get("attempts", 0)),
                "error_type": type(error).__name__,
                "error": str(error)[:500],
            },
        )


register_middleware(
    "app_insights",
    lambda options: AppInsightsMiddleware(
        connection_string=options.get("connection_string"),
        connection_string_env=options.get("connection_string_env"),
        instrumentation_key=options.get("instrumentation_key"),
        instrumentation_key_env=options.get("instrumentation_key_env"),
        role_name=options.get("role_name"),
        role_instance=options.get("role_instance"),
        enable_batch_events=options.get("enable_batch_events", True),
        enable_llm_events=options.get("enable_llm_events", False),
        enable_dependency_tracking=options.get("enable_dependency_tracking", True),
        on_error=options.get("on_error", "skip"),
    ),
    schema=_APPINSIGHTS_SCHEMA,
)


__all__ = ["AppInsightsMiddleware"]
