"""Result aggregation and payload building for SDA execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.core.controls import CostTracker
    from elspeth.core.sda.plugins import AggregationTransform


class ResultAggregator:
    """Aggregates results, failures, and metadata into final payload."""

    def __init__(
        self,
        aggregation_plugins: list[AggregationTransform],
        cost_tracker: CostTracker | None = None,
    ):
        """
        Initialize result aggregator.

        Args:
            aggregation_plugins: Aggregation transform plugins
            cost_tracker: Cost tracker for summary
        """
        self.aggregation_plugins = aggregation_plugins
        self.cost_tracker = cost_tracker
        self._results_with_index: list[tuple[int, dict[str, Any]]] = []
        self._failures: list[dict[str, Any]] = []

    def add_result(self, record: dict[str, Any], row_index: int) -> None:
        """Add successful result."""
        self._results_with_index.append((row_index, record))

    def add_failure(self, failure: dict[str, Any]) -> None:
        """Add failed result."""
        self._failures.append(failure)

    def build_payload(
        self,
        security_level: str | None = None,
        early_stop_reason: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build final payload with results, metadata, and aggregates.

        Args:
            security_level: Security level for metadata
            early_stop_reason: Early stop reason if triggered

        Returns:
            Complete payload dictionary
        """
        # Sort results by original index
        self._results_with_index.sort(key=lambda item: item[0])
        results = [record for _, record in self._results_with_index]

        # Build base payload
        payload: dict[str, Any] = {"results": results}
        if self._failures:
            payload["failures"] = self._failures

        # Apply aggregation plugins
        aggregates: dict[str, Any] = {}
        for plugin in self.aggregation_plugins:
            # Pass aggregates dict to plugin (growing dict)
            derived = plugin.aggregate(results, aggregates)

            # Validate output if plugin has schemas (skip validation only if None)
            if derived is not None and (hasattr(plugin, 'output_schema') or hasattr(plugin, 'input_schema')):
                from collections.abc import Mapping

                from elspeth.core.validation import ConfigurationError, validate_schema

                output_schema = getattr(plugin, 'output_schema', getattr(plugin, 'input_schema', None))
                if output_schema and isinstance(derived, Mapping):
                    errors = list(validate_schema(derived, output_schema))
                    if errors:
                        plugin_name = getattr(plugin, 'name', plugin.__class__.__name__)
                        error_messages = "\n".join(f"  - {e.format()}" for e in errors)
                        raise ConfigurationError(
                            f"Output validation failed for {plugin_name}:\n"
                            f"{error_messages}\n"
                            f"Expected schema: {output_schema}\n"
                            f"Actual data: {derived}"
                        )

            if derived is not None:
                # Use output_key if specified, otherwise plugin name
                output_key = plugin.config.get('output_key', plugin.name)
                aggregates[output_key] = derived

        if aggregates:
            payload["aggregates"] = aggregates

        # Build metadata
        metadata: dict[str, Any] = {
            "rows": len(results),
            "row_count": len(results),
        }

        # Add retry summary
        retry_summary = self._build_retry_summary(results, self._failures)
        if retry_summary:
            metadata["retry_summary"] = retry_summary

        # Add aggregates to metadata
        if aggregates:
            metadata["aggregates"] = aggregates

        # Add cost summary
        if self.cost_tracker:
            cost_summary = self.cost_tracker.summary()
            if cost_summary:
                payload["cost_summary"] = cost_summary
                metadata["cost_summary"] = cost_summary

        # Add failures count
        if self._failures:
            metadata["failures"] = self._failures

        # Add security level
        if security_level:
            metadata["security_level"] = security_level

        # Add early stop info
        if early_stop_reason:
            metadata["early_stop"] = early_stop_reason
            payload["early_stop"] = early_stop_reason

        payload["metadata"] = metadata
        return payload

    def _build_retry_summary(
        self,
        results: list[dict[str, Any]],
        failures: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Build retry summary from results and failures."""
        retry_summary = {
            "total_requests": len(results) + len(failures),
            "total_retries": 0,
            "exhausted": len(failures),
        }

        retry_present = False

        # Count retries in successful results
        for record in results:
            info = record.get("retry")
            if info:
                retry_present = True
                attempts = int(info.get("attempts", 1))
                retry_summary["total_retries"] += max(attempts - 1, 0)

        # Count retries in failures
        for failure in failures:
            info = failure.get("retry")
            if info:
                retry_present = True
                attempts = int(info.get("attempts", 0))
                retry_summary["total_retries"] += max(attempts - 1, 0)

        return retry_summary if retry_present else None
