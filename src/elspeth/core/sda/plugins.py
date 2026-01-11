"""SDA plugin interfaces for data transformation."""

from __future__ import annotations

from typing import Any, Protocol


class TransformPlugin(Protocol):
    """Transforms a single data input during the DECIDE phase.

    Transform plugins receive row data and a context dict for inter-plugin
    communication. They return the (potentially modified) row data.

    The context dict is a scratchpad that persists across all plugins for
    a single row. Plugins can read/write to context to communicate with
    downstream plugins in the chain.
    """

    name: str

    def transform(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]: ...


class AggregationTransform(Protocol):
    """Performs aggregation transformation across all results after individual transforms complete.

    The aggregates dict is passed to enable meta-plugin execution - plugins can read
    outputs from earlier plugins via input_key and write to custom keys via output_key.
    """

    name: str
    config: dict[str, Any]  # Required for output_key/input_key configuration

    def aggregate(
        self,
        records: list[dict[str, Any]],
        aggregates: dict[str, Any]
    ) -> dict[str, Any] | list[dict[str, Any]]: ...


class ComparisonPlugin(Protocol):
    """Compares variant payloads against baseline payload."""

    name: str

    def compare(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]: ...


class HaltConditionPlugin(Protocol):
    """Observes row-level results and signals when processing should halt."""

    name: str

    def reset(self) -> None: ...

    def check(self, record: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None: ...
