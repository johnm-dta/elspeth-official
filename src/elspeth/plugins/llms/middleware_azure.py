"""Azure environment middleware for telemetry and request tracing."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterable, Mapping
from typing import Any

from elspeth.core.llm.middleware import LLMMiddleware, LLMRequest
from elspeth.core.llm.registry import register_middleware

logger = logging.getLogger(__name__)


_AZURE_ENV_SCHEMA = {
    "type": "object",
    "properties": {
        "enable_run_logging": {"type": "boolean"},
        "log_prompts": {"type": "boolean"},
        "log_config_diffs": {"type": "boolean"},
        "log_metrics": {"type": "boolean"},
        "severity_threshold": {"type": "string"},
        "on_error": {"type": "string", "enum": ["abort", "skip"]},
    },
    "additionalProperties": True,
}

_SEVERITIES = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_AZURE_ENV_VARS = (
    "AZUREML_RUN_ID",
    "AZUREML_RUN_TOKEN",
    "AZUREML_ARM_SUBSCRIPTION",
    "AZUREML_ARM_RESOURCEGROUP",
    "AZUREML_ARM_WORKSPACE_NAME",
    "AML_APP_ROOT",
)


def _is_probably_running_in_azure() -> bool:
    """Heuristically detect whether the current process is inside Azure ML."""

    for name in _AZURE_ENV_VARS:
        value = os.getenv(name)
        if value:
            return True
    return False


def _resolve_azure_run() -> Any | None:
    """Attempt to resolve an Azure ML Run context."""

    try:
        from azureml.core import Run  # type: ignore
    except Exception:  # pragma: no cover - optional dependency guard
        return None

    try:
        run = Run.get_context(allow_offline=False)
    except Exception:  # pragma: no cover - runtime guard
        return None
    if getattr(run, "id", None):
        return run
    return None


class AzureEnvironmentMiddleware(LLMMiddleware):
    """Capture request/response telemetry for Azure ML environments."""

    name = "azure_environment"

    def __init__(
        self,
        *,
        enable_run_logging: bool = True,
        log_prompts: bool = False,
        log_config_diffs: bool = True,
        log_metrics: bool = True,
        severity_threshold: str = "INFO",
        on_error: str = "skip",
    ) -> None:
        if on_error not in {"abort", "skip"}:
            raise ValueError("on_error must be 'abort' or 'skip'")
        self.on_error = on_error
        self.log_prompts = log_prompts
        self.log_config_diffs = log_config_diffs
        self.log_metrics = log_metrics
        level_name = severity_threshold.upper()
        self._fallback_level = _SEVERITIES.get(level_name, logging.INFO)
        if level_name not in _SEVERITIES:
            logger.warning("Unknown severity_threshold '%s'; defaulting to INFO", severity_threshold)

        env_detected = _is_probably_running_in_azure()
        self._run = _resolve_azure_run() if enable_run_logging else None
        if enable_run_logging and self._run is None:
            if env_detected:
                message = (
                    "AzureEnvironmentMiddleware requires an Azure ML run context; " "ensure azureml-core is installed and the run is active"
                )
                if self.on_error == "skip":
                    logger.warning("%s. Continuing without run context due to on_error=skip", message)
                else:
                    raise RuntimeError(message)
            else:
                message = "Azure ML environment variables not detected; telemetry logging will be disabled."
                if self.on_error == "skip":
                    logger.info(message)
                else:
                    raise RuntimeError("AzureEnvironmentMiddleware requires an Azure ML run context when on_error=abort")
        self._inflight: dict[str, float] = {}
        self._lock = threading.Lock()
        self._sequence = 0
        self._suite_logged = False
        self._summary = {
            "experiments": 0,
            "total_rows": 0,
            "total_failures": 0,
        }
        if self._run:
            logger.info("AzureEnvironmentMiddleware linked to run %s", getattr(self._run, "id", "unknown"))
        else:
            logger.log(self._fallback_level, "AzureEnvironmentMiddleware running without Azure ML run context")

    def before_request(self, request: LLMRequest) -> LLMRequest:
        sequence = self._next_sequence()
        timestamp = time.time()
        with self._lock:
            self._inflight[sequence] = timestamp
        payload = {
            "sequence": sequence,
            "timestamp": timestamp,
        }
        payload.update(request.metadata)
        if self.log_prompts:
            payload["system_prompt"] = request.system_prompt
            payload["user_prompt"] = request.user_prompt
        self._log_row("llm_request", payload)
        updated_metadata = dict(request.metadata)
        updated_metadata["azure_sequence"] = sequence
        return request.clone(metadata=updated_metadata)

    def after_response(self, request: LLMRequest, response: dict[str, Any]) -> dict[str, Any]:
        sequence = request.metadata.get("azure_sequence")
        duration = None
        if sequence is not None:
            with self._lock:
                start = self._inflight.pop(sequence, None)
            if start is not None:
                duration = max(time.time() - start, 0.0)
        payload = {
            "sequence": sequence,
            "timestamp": time.time(),
            "duration": duration,
        }
        payload.update(request.metadata)
        metrics = response.get("metrics") if isinstance(response, dict) else None
        if self.log_metrics and isinstance(metrics, dict):
            for key, value in metrics.items():
                payload[f"metric_{key}"] = value
        error = response.get("error") if isinstance(response, dict) else None
        if error:
            payload["error"] = error
        self._log_row("llm_response", payload)
        return response

    # ---- Suite lifecycle -------------------------------------------------

    def on_suite_loaded(
        self,
        experiments: Iterable[Mapping[str, Any]],
        preflight: Mapping[str, Any] | None = None,
    ) -> None:
        if self._suite_logged:
            return
        experiments = list(experiments)
        self._suite_logged = True
        self._log_metric("experiment_count", len(experiments))
        for entry in experiments:
            self._log_row("experiments", dict(entry))
        if preflight:
            self._log_row("suite_preflight", dict(preflight))

    def on_experiment_start(self, name: str, metadata: Mapping[str, Any]) -> None:
        payload = {
            "experiment": name,
            "timestamp": time.time(),
        }
        payload.update(metadata)
        self._log_row("experiment_start", payload)

    def on_experiment_complete(
        self,
        name: str,
        payload: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        results = payload.get("results") if isinstance(payload, Mapping) else None
        failures = payload.get("failures") if isinstance(payload, Mapping) else None
        aggregates = payload.get("aggregates") if isinstance(payload, Mapping) else None
        cost_summary = payload.get("cost_summary") if isinstance(payload, Mapping) else None

        rows = len(results) if isinstance(results, list) else 0
        failure_count = len(failures) if isinstance(failures, list) else 0

        self._summary["experiments"] += 1
        self._summary["total_rows"] += rows
        self._summary["total_failures"] += failure_count

        record = {
            "experiment": name,
            "timestamp": time.time(),
            "rows": rows,
            "failures": failure_count,
        }
        if metadata:
            record.update(metadata)
        if cost_summary:
            for key, value in cost_summary.items():
                record[f"cost_{key}"] = value
        if aggregates:
            record["aggregates"] = json.dumps(aggregates)
            if self.log_metrics:
                self._log_table(
                    f"experiment_{name}_aggregates",
                    {
                        key: [value] if not isinstance(value, Mapping) else [json.dumps(value, sort_keys=True)]
                        for key, value in aggregates.items()
                    },
                )
        if failures:
            record["failures_detail"] = json.dumps(failures[:3])  # sample failures
        self._log_row("experiment_complete", record)

    def on_baseline_comparison(self, experiment: str, comparison: Mapping[str, Any]) -> None:
        if not comparison or not self.log_config_diffs:
            return
        for plugin, diff in comparison.items():
            table_name = f"baseline_{experiment}_{plugin}"
            self._log_table(table_name, {key: [value] for key, value in diff.items()})

    def on_suite_complete(self) -> None:
        summary = dict(self._summary)
        summary["timestamp"] = time.time()
        self._log_row("suite_summary", summary)

    def on_retry_exhausted(self, request, metadata, error) -> None:  # type: ignore[override]
        payload = {
            "timestamp": time.time(),
            "sequence": request.metadata.get("azure_sequence"),
            "attempts": metadata.get("attempts"),
            "max_attempts": metadata.get("max_attempts"),
            "error": metadata.get("error"),
            "error_type": metadata.get("error_type"),
        }
        if metadata.get("history"):
            payload["history"] = json.dumps(metadata["history"])
        self._log_row("llm_retry_exhausted", payload)

    def _log_row(self, name: str, payload: dict[str, Any]) -> None:
        if not self._run:
            logger.log(self._fallback_level, "[azure_env] %s: %s", name, payload)
            return
        try:
            self._run.log_row(name, **payload)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.log(
                self._fallback_level,
                "Azure ML run log_row failed: %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            logger.log(self._fallback_level, "[azure_env-fallback] %s: %s", name, payload)

    def _log_table(self, name: str, payload: Mapping[str, Any]) -> None:
        if not self._run:
            logger.log(self._fallback_level, "[azure_env-table] %s: %s", name, payload)
            return
        try:
            self._run.log_table(name, payload)
        except Exception as exc:  # pragma: no cover - telemetry best effort
            logger.log(
                self._fallback_level,
                "Azure ML run log_table failed: %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            logger.log(self._fallback_level, "[azure_env-table-fallback] %s: %s", name, payload)

    def _log_metric(self, name: str, value: Any) -> None:
        if not self._run:
            logger.log(self._fallback_level, "[azure_env-metric] %s=%s", name, value)
            return
        try:
            self._run.log(name, value)
        except Exception as exc:  # pragma: no cover - telemetry best effort
            logger.log(
                self._fallback_level,
                "Azure ML run log failed: %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            logger.log(self._fallback_level, "[azure_env-metric-fallback] %s=%s", name, value)

    def _next_sequence(self) -> str:
        with self._lock:
            self._sequence += 1
            return f"az-{self._sequence}"


register_middleware(
    "azure_environment",
    lambda options: AzureEnvironmentMiddleware(
        enable_run_logging=options.get("enable_run_logging", True),
        log_prompts=options.get("log_prompts", False),
        log_config_diffs=options.get("log_config_diffs", True),
        log_metrics=options.get("log_metrics", True),
        severity_threshold=options.get("severity_threshold", "INFO"),
        on_error=options.get("on_error", "skip"),
    ),
    schema=_AZURE_ENV_SCHEMA,
)


__all__ = ["AzureEnvironmentMiddleware"]
