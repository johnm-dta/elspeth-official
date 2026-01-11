"""Distribution testing analyzer plugins (collection-mode)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from elspeth.core.sda.plugin_registry import register_comparison_plugin

from .stats_utils import ON_ERROR_SCHEMA, collect_scores_by_criterion, scipy_stats

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


_DISTRIBUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "input_key": {"type": "string", "description": "Collection key in aggregates (optional for comparison use)"},
        "source_field": {"type": "string"},
        "criteria": {"type": "array", "items": {"type": "string"}},
        "min_samples": {"type": "integer", "minimum": 2},
        "on_error": ON_ERROR_SCHEMA,
    },
    "additionalProperties": True,
}


def _compute_distribution_shift(
    baseline: Sequence[float],
    variant: Sequence[float],
) -> dict[str, Any]:
    """Compute distribution shift metrics between baseline and variant.

    Args:
        baseline: Baseline scores
        variant: Variant scores

    Returns:
        Dict with KS test, Mann-Whitney U, and JS divergence
    """
    arr_base = np.asarray(list(baseline), dtype=float)
    arr_var = np.asarray(list(variant), dtype=float)
    n_base = arr_base.size
    n_var = arr_var.size
    mean_base = float(arr_base.mean()) if n_base else 0.0
    mean_var = float(arr_var.mean()) if n_var else 0.0
    var_base = float(arr_base.var(ddof=1)) if n_base > 1 else 0.0
    var_var = float(arr_var.var(ddof=1)) if n_var > 1 else 0.0
    std_base = float(np.sqrt(var_base)) if var_base > 0 else 0.0
    std_var = float(np.sqrt(var_var)) if var_var > 0 else 0.0

    ks_stat = None
    ks_pvalue = None
    if scipy_stats is not None and n_base >= 2 and n_var >= 2:
        try:
            ks = scipy_stats.ks_2samp(arr_base, arr_var, alternative="two-sided")
            ks_stat = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)
        except Exception:  # pragma: no cover
            ks_stat = None
            ks_pvalue = None

    mw_stat = None
    mw_pvalue = None
    if scipy_stats is not None and n_base >= 2 and n_var >= 2:
        try:
            mw = scipy_stats.mannwhitneyu(arr_base, arr_var, alternative="two-sided")
            mw_stat = float(mw.statistic)
            mw_pvalue = float(mw.pvalue)
        except Exception:  # pragma: no cover
            mw_stat = None
            mw_pvalue = None

    # Jensen-Shannon divergence with smoothing
    try:
        hist_range = (
            float(min(arr_base.min(initial=0), arr_var.min(initial=0))),
            float(max(arr_base.max(initial=0), arr_var.max(initial=0))),
        )
        if hist_range[0] == hist_range[1]:
            js_divergence = 0.0
        else:
            hist_base, bins = np.histogram(arr_base, bins="auto", range=hist_range, density=True)
            hist_var, _ = np.histogram(arr_var, bins=bins, density=True)
            hist_base = hist_base + 1e-12
            hist_var = hist_var + 1e-12
            hist_base /= hist_base.sum()
            hist_var /= hist_var.sum()
            m = 0.5 * (hist_base + hist_var)
            js_divergence = float(0.5 * (np.sum(hist_base * np.log(hist_base / m)) + np.sum(hist_var * np.log(hist_var / m))))
    except Exception:  # pragma: no cover
        js_divergence = None

    return {
        "baseline_samples": n_base,
        "variant_samples": n_var,
        "baseline_mean": mean_base,
        "variant_mean": mean_var,
        "baseline_std": std_base,
        "variant_std": std_var,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "mwu_statistic": mw_stat,
        "mwu_pvalue": mw_pvalue,
        "js_divergence": js_divergence,
    }


class ScoreDistributionAnalyzer:
    """Assess distribution properties from FieldCollector collection.

    Note: This analyzer can work with collections OR as a comparison plugin
    for baseline/variant comparisons.
    """

    name = "score_distribution"
    config_schema: ClassVar[dict[str, Any]] = _DISTRIBUTION_SCHEMA

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "collection",
        "description": "Collection with score data"
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Distribution test results"
    }

    def __init__(
        self,
        *,
        input_key: str,
        source_field: str = "score",
        criteria: Sequence[str] | None = None,
        min_samples: int = 2,
        on_error: str = "abort",
    ) -> None:
        """Initialize analyzer.

        Args:
            input_key: Key to read collection from aggregates dict
            source_field: Field name containing scores in collection
            criteria: Optional list of criteria to analyze
            min_samples: Minimum sample size
            on_error: Error handling mode ("abort" or "skip")
        """
        self.input_key = input_key
        self.source_field = source_field
        self._criteria = set(criteria) if criteria else None
        self._min_samples = max(int(min_samples), 2)
        if on_error not in {"abort", "skip"}:
            raise ValueError("on_error must be 'abort' or 'skip'")
        self._on_error = on_error
        self.config = {
            "input_key": input_key,
            "source_field": source_field,
            "criteria": criteria,
            "min_samples": min_samples,
            "on_error": on_error,
        }

    def aggregate(self, records: list[dict[str, Any]], aggregates: dict[str, Any]) -> dict[str, Any]:
        # This analyzer can be used as comparison plugin, so aggregate returns empty
        return {}

    def compare(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._compare_impl(baseline, variant)
        except Exception as exc:  # pragma: no cover - defensive guard
            if self._on_error == "skip":
                logger.warning("score_distribution skipped due to error: %s", exc)
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
            base = base_scores.get(name, [])
            var = var_scores.get(name, [])
            if len(base) < self._min_samples or len(var) < self._min_samples:
                continue
            stats = _compute_distribution_shift(base, var)
            if stats:
                results[name] = stats
        return results


# Backward compatibility alias
ScoreDistributionAggregator = ScoreDistributionAnalyzer


register_comparison_plugin(
    "score_distribution",
    lambda options: ScoreDistributionAnalyzer(
        input_key=options.get("input_key", ""),  # May not be needed for comparison use
        source_field=options.get("source_field", "score"),
        criteria=options.get("criteria"),
        min_samples=int(options.get("min_samples", 2)),
        on_error=options.get("on_error", "abort"),
    ),
    schema=_DISTRIBUTION_SCHEMA,
)
