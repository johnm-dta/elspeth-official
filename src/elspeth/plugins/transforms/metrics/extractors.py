"""Score extraction plugins (row-mode)."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from elspeth.core.sda.plugin_registry import register_transform_plugin

from .stats_utils import ON_ERROR_SCHEMA

logger = logging.getLogger(__name__)


_EXTRACTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "criteria": {"type": "array", "items": {"type": "string"}},
        "parse_json_content": {"type": "boolean"},
        "allow_missing": {"type": "boolean"},
        "threshold": {"type": "number"},
        "threshold_mode": {"type": "string", "enum": ["gt", "gte", "lt", "lte"]},
        "flag_field": {"type": "string"},
        "on_error": ON_ERROR_SCHEMA,
    },
    "additionalProperties": True,
}


class ScoreExtractorPlugin:
    """Extract numeric scores from LLM responses.

    The plugin inspects the per-criteria response payload for numeric values under
    the configured key (default: ``score``). Values are normalised to ``float``
    whenever possible. When ``threshold`` is supplied the plugin also flags rows
    that meet the threshold for downstream aggregators.
    """

    name = "score_extractor"

    # Config schema
    config_schema: ClassVar[dict[str, Any]] = _EXTRACTOR_SCHEMA

    # Input schema: accepts row object (any structure - plugin extracts from responses)
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with criteria response data"
    }

    # Output schema: row object with extracted score fields
    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with extracted numeric scores",
        "properties": {
            # Dynamic properties based on criteria and key config
            # Plugin adds fields like: scores, score_flags
        }
    }

    def __init__(
        self,
        *,
        key: str = "score",
        criteria: list[str] | None = None,
        parse_json_content: bool = True,
        allow_missing: bool = False,
        threshold: float | None = None,
        threshold_mode: str = "gte",
        flag_field: str = "score_flags",
    ) -> None:
        self._key = key
        self._criteria = set(criteria) if criteria else None
        self._parse_json = parse_json_content
        self._allow_missing = allow_missing
        self._threshold = threshold
        self._threshold_mode = threshold_mode
        self._flag_field = flag_field

    def transform(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Extract scores from LLM responses in context.

        With the new architecture, LLM responses are stored in context
        by the llm_query plugin. This plugin reads from context to
        extract numeric scores.

        Args:
            row: Current row data
            context: Shared context containing LLM responses from llm_query

        Returns:
            Dict with extracted scores (merged into row by RowProcessor)
        """
        scores: dict[str, float] = {}
        flags: dict[str, bool] = {}

        # Iterate over context entries looking for LLM responses
        for key, response in context.items():
            if not isinstance(response, Mapping):
                continue
            if self._criteria and key not in self._criteria:
                continue
            value = self._extract_value(response)
            if value is None:
                if not self._allow_missing:
                    scores[key] = np.nan
                continue
            scores[key] = value
            if self._threshold is not None:
                flags[key] = self._compare_threshold(value)

        derived: dict[str, Any] = {}
        if scores:
            derived.setdefault("scores", {}).update(scores)
        if flags:
            derived[self._flag_field] = flags
        return derived

    def _extract_value(self, response: Mapping[str, Any]) -> float | None:
        metrics = response.get("metrics") if isinstance(response, Mapping) else None
        if isinstance(metrics, Mapping) and self._key in metrics:
            return self._coerce_number(metrics.get(self._key))

        if self._parse_json:
            content = response.get("content") if isinstance(response, Mapping) else None
            if isinstance(content, str):
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, Mapping) and self._key in payload:
                    return self._coerce_number(payload.get(self._key))
        return None

    @staticmethod
    def _coerce_number(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.endswith("%"):
                cleaned = cleaned[:-1]
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _compare_threshold(self, value: float) -> bool:
        mode = self._threshold_mode
        if self._threshold is None:
            raise ValueError("threshold is None")
        threshold = float(self._threshold)
        if mode == "gt":
            return value > threshold
        if mode == "gte":
            return value >= threshold
        if mode == "lt":
            return value < threshold
        if mode == "lte":
            return value <= threshold
        raise ValueError(f"Unsupported threshold_mode '{mode}'")


register_transform_plugin(
    "score_extractor",
    lambda options: ScoreExtractorPlugin(
        key=options.get("key", "score"),
        criteria=options.get("criteria"),
        parse_json_content=options.get("parse_json_content", True),
        allow_missing=options.get("allow_missing", False),
        threshold=options.get("threshold"),
        threshold_mode=options.get("threshold_mode", "gte"),
        flag_field=options.get("flag_field", "score_flags"),
    ),
    schema=_EXTRACTOR_SCHEMA,
)
