"""Practical significance plugins (row-mode)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from elspeth.core.sda.plugin_registry import register_comparison_plugin

from .stats_utils import ON_ERROR_SCHEMA, collect_paired_scores_by_criterion

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


_PRACTICAL_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {"type": "array", "items": {"type": "string"}},
        "threshold": {"type": "number"},
        "success_threshold": {"type": "number"},
        "min_samples": {"type": "integer", "minimum": 1},
        "on_error": ON_ERROR_SCHEMA,
    },
    "additionalProperties": True,
}


class ScorePracticalBaselinePlugin:
    """Assess practical significance (meaningful change, NNT, success deltas)."""

    name = "score_practical"
    config_schema: ClassVar[dict[str, Any]] = _PRACTICAL_SCHEMA

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with score arrays and success threshold"
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with success rates and practical significance"
    }

    def __init__(
        self,
        *,
        criteria: Sequence[str] | None = None,
        threshold: float = 1.0,
        success_threshold: float = 4.0,
        min_samples: int = 1,
        on_error: str = "abort",
    ) -> None:
        self._criteria = set(criteria) if criteria else None
        self._threshold = float(threshold)
        self._success_threshold = float(success_threshold)
        self._min_samples = max(int(min_samples), 1)
        if on_error not in {"abort", "skip"}:
            raise ValueError("on_error must be 'abort' or 'skip'")
        self._on_error = on_error

    def compare(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._compare_impl(baseline, variant)
        except Exception as exc:  # pragma: no cover
            if self._on_error == "skip":
                logger.warning("score_practical skipped due to error: %s", exc)
                return {}
            raise

    def _compare_impl(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        pairs = collect_paired_scores_by_criterion(baseline, variant)
        criteria = sorted(pairs.keys())
        if self._criteria is not None:
            criteria = [name for name in criteria if name in self._criteria]

        results: dict[str, Any] = {}
        for name in criteria:
            paired = pairs.get(name, [])
            if len(paired) < self._min_samples:
                continue
            diffs = [v - b for b, v in paired]
            meaningful = [abs(d) >= self._threshold for d in diffs]
            meaningful_rate = sum(meaningful) / len(paired)
            baseline_success = sum(1 for b, _ in paired if b >= self._success_threshold) / len(paired)
            variant_success = sum(1 for _, v in paired if v >= self._success_threshold) / len(paired)
            success_delta = variant_success - baseline_success
            nnt = 1.0 / success_delta if success_delta > 0 else float("inf")
            results[name] = {
                "pairs": len(paired),
                "mean_difference": float(np.mean(diffs)),
                "median_difference": float(np.median(diffs)),
                "meaningful_change_rate": meaningful_rate,
                "success_threshold": self._success_threshold,
                "baseline_success_rate": baseline_success,
                "variant_success_rate": variant_success,
                "success_delta": success_delta,
                "number_needed_to_treat": nnt,
            }
        return results


register_comparison_plugin(
    "score_practical",
    lambda options: ScorePracticalBaselinePlugin(
        criteria=options.get("criteria"),
        threshold=float(options.get("threshold", 1.0)),
        success_threshold=float(options.get("success_threshold", 4.0)),
        min_samples=int(options.get("min_samples", 1)),
        on_error=options.get("on_error", "abort"),
    ),
    schema=_PRACTICAL_SCHEMA,
)
