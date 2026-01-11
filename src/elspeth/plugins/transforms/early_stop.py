"""Default early-stop plugin implementations."""

from __future__ import annotations

from typing import Any

from elspeth.core.sda.plugin_registry import register_halt_condition_plugin


class ThresholdEarlyStopPlugin:
    """Stops execution when a metric crosses a configured threshold."""

    name = "threshold"

    def __init__(
        self,
        *,
        metric: str,
        threshold: Any,
        comparison: str = "gte",
        min_rows: int = 1,
        label: str | None = None,
    ) -> None:
        if not metric:
            raise ValueError("Threshold early-stop plugin requires a 'metric' path")
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid threshold value {threshold!r}") from exc

        comparison_key = str(comparison or "gte").lower()
        if comparison_key not in {"gte", "gt", "lte", "lt"}:
            comparison_key = "gte"

        self._metric = metric
        self._threshold = threshold_value
        self._comparison = comparison_key
        self._min_rows = max(int(min_rows or 1), 1)
        self._label = label
        self._rows_observed = 0
        self._triggered_reason: dict[str, Any] | None = None

    def reset(self) -> None:
        self._rows_observed = 0
        self._triggered_reason = None

    def check(self, record: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if self._triggered_reason:
            return dict(self._triggered_reason)

        metrics = record.get("metrics") or {}
        value = self._extract_metric(metrics, self._metric)
        if value is None:
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        self._rows_observed += 1
        if self._rows_observed < self._min_rows:
            return None

        if not self._compare(numeric_value, self._threshold, self._comparison):
            return None

        reason: dict[str, Any] = {
            "metric": self._metric,
            "comparison": self._comparison,
            "threshold": self._threshold,
            "value": numeric_value,
            "rows_observed": self._rows_observed,
        }
        if self._label:
            reason["label"] = self._label
        if metadata:
            reason.update({k: v for k, v in metadata.items() if k not in reason})
        self._triggered_reason = reason
        return dict(reason)

    @staticmethod
    def _extract_metric(metrics: dict[str, Any], name: str) -> Any:
        current: Any = metrics
        for part in name.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    @staticmethod
    def _compare(value: float, threshold: float, comparison: str) -> bool:
        if comparison == "gt":
            return value > threshold
        if comparison == "lte":
            return value <= threshold
        if comparison == "lt":
            return value < threshold
        return value >= threshold


register_halt_condition_plugin(
    ThresholdEarlyStopPlugin.name,
    lambda options: ThresholdEarlyStopPlugin(
        metric=options.get("metric", ""),
        threshold=options.get("threshold"),
        comparison=options.get("comparison", "gte"),
        min_rows=options.get("min_rows", 1),
        label=options.get("label"),
    ),
    schema={
        "type": "object",
        "properties": {
            "metric": {"type": "string", "minLength": 1},
            "threshold": {"type": ["number", "string"]},
            "comparison": {"type": "string", "enum": ["gte", "gt", "lte", "lt"]},
            "min_rows": {"type": "integer", "minimum": 1},
            "label": {"type": "string"},
        },
        "required": ["metric", "threshold"],
        "additionalProperties": True,
    },
)
