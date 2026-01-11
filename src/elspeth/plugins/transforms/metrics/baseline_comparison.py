"""Baseline comparison plugins (row-mode): delta, Cliff's delta, assumptions."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from elspeth.core.sda.plugin_registry import register_comparison_plugin

from .stats_utils import ON_ERROR_SCHEMA, calculate_cliffs_delta, collect_scores_by_criterion, scipy_stats

logger = logging.getLogger(__name__)


_DELTA_SCHEMA = {
    "type": "object",
    "properties": {
        "metric": {"type": "string"},
        "criteria": {"type": "array", "items": {"type": "string"}},
        "on_error": ON_ERROR_SCHEMA,
    },
    "additionalProperties": True,
}

_CLIFFS_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {"type": "array", "items": {"type": "string"}},
        "min_samples": {"type": "integer", "minimum": 1},
        "on_error": ON_ERROR_SCHEMA,
    },
    "additionalProperties": True,
}

_ASSUMPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {"type": "array", "items": {"type": "string"}},
        "min_samples": {"type": "integer", "minimum": 3},
        "alpha": {"type": "number", "minimum": 0.001, "maximum": 0.2},
        "on_error": ON_ERROR_SCHEMA,
    },
    "additionalProperties": True,
}


class ScoreDeltaBaselinePlugin:
    """Compare score statistics between baseline and variant."""

    name = "score_delta"
    config_schema: ClassVar[dict[str, Any]] = _DELTA_SCHEMA

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with score and baseline fields"
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with delta and ratio fields added"
    }

    def __init__(self, *, metric: str = "mean", criteria: list[str] | None = None) -> None:
        self._metric = metric
        # criteria parameter kept for backward compatibility but unused with new flat schema
        self._criteria = set(criteria) if criteria else None

    def compare(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        base_stats = self._extract_stats(baseline)
        var_stats = self._extract_stats(variant)
        if not base_stats or not var_stats:
            return {}

        # Extract metric values from flat schema
        base_metric = base_stats.get(self._metric)
        var_metric = var_stats.get(self._metric)
        if base_metric is None or var_metric is None:
            return {}

        # Return delta and ratio
        delta = var_metric - base_metric
        ratio = var_metric / base_metric if base_metric != 0 else None

        return {
            "delta": delta,
            "ratio": ratio,
            "baseline_value": base_metric,
            "variant_value": var_metric,
        }

    @staticmethod
    def _extract_stats(payload: dict[str, Any]) -> dict[str, Any]:
        aggregates = payload.get("aggregates") if isinstance(payload, Mapping) else None
        if not isinstance(aggregates, Mapping):
            return {}
        stats = aggregates.get("score_stats")
        if not isinstance(stats, Mapping):
            return {}
        # Return flat stats schema (no more "criteria" nesting)
        return dict(stats)


class ScoreCliffsDeltaPlugin:
    """Compute Cliff's delta effect size between baseline and variant."""

    name = "score_cliffs_delta"
    config_schema: ClassVar[dict[str, Any]] = _CLIFFS_SCHEMA

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with treatment and baseline score arrays"
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with cliffs_delta and effect_size_category fields"
    }

    def __init__(self, *, criteria: Sequence[str] | None = None, min_samples: int = 1, on_error: str = "abort") -> None:
        self._criteria = set(criteria) if criteria else None
        self._min_samples = max(int(min_samples), 1)
        if on_error not in {"abort", "skip"}:
            raise ValueError("on_error must be 'abort' or 'skip'")
        self._on_error = on_error

    def compare(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._compare_impl(baseline, variant)
        except Exception as exc:  # pragma: no cover - defensive guard
            if self._on_error == "skip":
                logger.warning("score_cliffs_delta skipped due to error: %s", exc)
                return {}
            raise

    def _compare_impl(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        base_scores = collect_scores_by_criterion(baseline)
        var_scores = collect_scores_by_criterion(variant)
        criteria = sorted(set(base_scores.keys()) & set(var_scores.keys()))
        if self._criteria is not None:
            criteria = [name for name in criteria if name in self._criteria]

        results: dict[str, Any] = {}
        for name in criteria:
            group1 = base_scores.get(name, [])
            group2 = var_scores.get(name, [])
            if len(group1) < self._min_samples or len(group2) < self._min_samples:
                continue
            delta, interpretation = calculate_cliffs_delta(group1, group2)
            results[name] = {
                "delta": delta,
                "interpretation": interpretation,
                "baseline_samples": len(group1),
                "variant_samples": len(group2),
            }
        return results


class ScoreAssumptionsBaselinePlugin:
    """Report normality and variance diagnostics for baseline vs. variant scores."""

    name = "score_assumptions"
    config_schema: ClassVar[dict[str, Any]] = _ASSUMPTION_SCHEMA

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with treatment and baseline score arrays"
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with normality and variance test results"
    }

    def __init__(
        self,
        *,
        criteria: Sequence[str] | None = None,
        min_samples: int = 3,
        alpha: float = 0.05,
        on_error: str = "abort",
    ) -> None:
        self._criteria = set(criteria) if criteria else None
        self._min_samples = max(int(min_samples), 3)
        self._alpha = float(alpha)
        if on_error not in {"abort", "skip"}:
            raise ValueError("on_error must be 'abort' or 'skip'")
        self._on_error = on_error

    def compare(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._compare_impl(baseline, variant)
        except Exception as exc:  # pragma: no cover
            if self._on_error == "skip":
                logger.warning("score_assumptions skipped due to error: %s", exc)
                return {}
            raise

    def _compare_impl(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        if scipy_stats is None:
            return {}
        base_scores = collect_scores_by_criterion(baseline)
        var_scores = collect_scores_by_criterion(variant)
        criteria = sorted(set(base_scores.keys()) & set(var_scores.keys()))
        if self._criteria is not None:
            criteria = [name for name in criteria if name in self._criteria]

        results: dict[str, Any] = {}
        for name in criteria:
            base = base_scores.get(name, [])
            var = var_scores.get(name, [])
            entry: dict[str, Any] = {}
            if len(base) >= self._min_samples:
                try:
                    stat, pval = scipy_stats.shapiro(base)
                    entry["baseline"] = {
                        "statistic": float(stat),
                        "p_value": float(pval),
                        "is_normal": bool(pval > self._alpha),
                        "samples": len(base),
                    }
                except Exception:
                    entry["baseline"] = None
            else:
                entry["baseline"] = None
            if len(var) >= self._min_samples:
                try:
                    stat, pval = scipy_stats.shapiro(var)
                    entry["variant"] = {
                        "statistic": float(stat),
                        "p_value": float(pval),
                        "is_normal": bool(pval > self._alpha),
                        "samples": len(var),
                    }
                except Exception:
                    entry["variant"] = None
            else:
                entry["variant"] = None
            if len(base) >= 2 and len(var) >= 2:
                try:
                    stat, pval = scipy_stats.levene(base, var)
                    entry["variance"] = {
                        "statistic": float(stat),
                        "p_value": float(pval),
                        "equal_variance": bool(pval > self._alpha),
                    }
                except Exception:
                    entry["variance"] = None
            else:
                entry["variance"] = None
            if any(entry.values()):
                results[name] = entry
        return results


register_comparison_plugin(
    "score_delta",
    lambda options: ScoreDeltaBaselinePlugin(
        metric=options.get("metric", "mean"),
        criteria=options.get("criteria"),
    ),
    schema=_DELTA_SCHEMA,
)

register_comparison_plugin(
    "score_cliffs_delta",
    lambda options: ScoreCliffsDeltaPlugin(
        criteria=options.get("criteria"),
        min_samples=int(options.get("min_samples", 1)),
        on_error=options.get("on_error", "abort"),
    ),
    schema=_CLIFFS_SCHEMA,
)

register_comparison_plugin(
    "score_assumptions",
    lambda options: ScoreAssumptionsBaselinePlugin(
        criteria=options.get("criteria"),
        min_samples=int(options.get("min_samples", 3)),
        alpha=float(options.get("alpha", 0.05)),
        on_error=options.get("on_error", "abort"),
    ),
    schema=_ASSUMPTION_SCHEMA,
)
