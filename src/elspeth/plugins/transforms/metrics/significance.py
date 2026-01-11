"""Statistical significance testing plugins (row-mode)."""

from __future__ import annotations

import logging
import math
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from elspeth.core.sda.plugin_registry import register_comparison_plugin

from .stats_utils import ON_ERROR_SCHEMA, benjamini_hochberg, collect_scores_by_criterion, scipy_stats

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


_SIGNIFICANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {"type": "array", "items": {"type": "string"}},
        "min_samples": {"type": "integer", "minimum": 2},
        "equal_var": {"type": "boolean"},
        "on_error": ON_ERROR_SCHEMA,
        "adjustment": {"type": "string", "enum": ["none", "bonferroni", "fdr"]},
        "family_size": {"type": "integer", "minimum": 1},
    },
    "additionalProperties": True,
}

_BAYESIAN_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {"type": "array", "items": {"type": "string"}},
        "min_samples": {"type": "integer", "minimum": 2},
        "credible_interval": {"type": "number", "minimum": 0.5, "maximum": 0.999},
        "on_error": ON_ERROR_SCHEMA,
    },
    "additionalProperties": True,
}


def _compute_significance(
    baseline: Sequence[float],
    variant: Sequence[float],
    *,
    equal_var: bool = False,
) -> dict[str, Any]:
    """Compute statistical significance metrics.

    Args:
        baseline: Baseline scores
        variant: Variant scores
        equal_var: Whether to assume equal variance

    Returns:
        Dict with test statistics, p-value, effect size, etc.
    """
    arr_base = np.asarray(list(baseline), dtype=float)
    arr_var = np.asarray(list(variant), dtype=float)
    n_base = arr_base.size
    n_var = arr_var.size
    mean_base = float(arr_base.mean()) if n_base else 0.0
    mean_var = float(arr_var.mean()) if n_var else 0.0
    mean_diff = mean_var - mean_base
    var_base = float(arr_base.var(ddof=1)) if n_base > 1 else 0.0
    var_var = float(arr_var.var(ddof=1)) if n_var > 1 else 0.0
    std_base = math.sqrt(var_base) if var_base > 0 else 0.0
    std_var = math.sqrt(var_var) if var_var > 0 else 0.0

    denom = math.sqrt((var_base / n_base if n_base > 0 else 0.0) + (var_var / n_var if n_var > 0 else 0.0))
    t_stat = mean_diff / denom if denom > 0 else None

    df: float | None = None

    if equal_var and n_base > 1 and n_var > 1:
        df = float(n_base + n_var - 2)
    else:
        term_base = (var_base / n_base) if n_base > 1 else 0.0
        term_var = (var_var / n_var) if n_var > 1 else 0.0
        denom_terms = term_base + term_var
        if denom_terms > 0:
            numerator = denom_terms**2
            denominator = 0.0
            if n_base > 1 and term_base > 0:
                denominator += (term_base**2) / (n_base - 1)
            if n_var > 1 and term_var > 0:
                denominator += (term_var**2) / (n_var - 1)
            df = numerator / denominator if denominator > 0 else None
        else:
            df = None

    pooled = None
    if n_base > 1 and n_var > 1:
        pooled = ((n_base - 1) * var_base + (n_var - 1) * var_var) / (n_base + n_var - 2)
    effect_size = None
    if pooled is not None and pooled > 0:
        effect_size = mean_diff / math.sqrt(pooled)
    elif std_base > 0 or std_var > 0:
        pooled_var = ((std_base**2) + (std_var**2)) / 2
        if pooled_var > 0:
            effect_size = mean_diff / math.sqrt(pooled_var)

    p_value = None
    if t_stat is not None and "df" in locals() and df is not None and scipy_stats is not None:
        try:
            p_value = float(scipy_stats.t.sf(abs(t_stat), df) * 2)
        except Exception:  # pragma: no cover - scipy failure
            p_value = None

    return {
        "baseline_mean": mean_base,
        "variant_mean": mean_var,
        "mean_difference": mean_diff,
        "baseline_std": std_base,
        "variant_std": std_var,
        "baseline_samples": n_base,
        "variant_samples": n_var,
        "effect_size": effect_size,
        "t_stat": t_stat,
        "degrees_of_freedom": df,
        "p_value": p_value,
    }


def _compute_bayesian_summary(
    baseline: Sequence[float],
    variant: Sequence[float],
    alpha: float,
) -> dict[str, Any]:
    """Compute Bayesian credible interval and probability.

    Args:
        baseline: Baseline scores
        variant: Variant scores
        alpha: Alpha level for credible interval

    Returns:
        Dict with Bayesian analysis results
    """
    arr_base = np.asarray(list(baseline), dtype=float)
    arr_var = np.asarray(list(variant), dtype=float)
    n_base = arr_base.size
    n_var = arr_var.size
    mean_base = float(arr_base.mean()) if n_base else 0.0
    mean_var = float(arr_var.mean()) if n_var else 0.0
    mean_diff = mean_var - mean_base
    var_base = float(arr_base.var(ddof=1)) if n_base > 1 else 0.0
    var_var = float(arr_var.var(ddof=1)) if n_var > 1 else 0.0
    stderr = math.sqrt((var_base / n_base if n_base > 0 else 0.0) + (var_var / n_var if n_var > 0 else 0.0))
    if stderr <= 0:
        return {}

    term_base = (var_base / n_base) if n_base > 1 else 0.0
    term_var = (var_var / n_var) if n_var > 1 else 0.0
    denom_terms = term_base + term_var
    df = None
    if denom_terms > 0:
        numerator = denom_terms**2
        denominator = 0.0
        if n_base > 1 and term_base > 0:
            denominator += (term_base**2) / (n_base - 1)
        if n_var > 1 and term_var > 0:
            denominator += (term_var**2) / (n_var - 1)
        df = numerator / denominator if denominator > 0 else None

    if df is not None and scipy_stats is not None:
        dist = scipy_stats.t(df, loc=mean_diff, scale=stderr)
        prob = 1 - float(dist.cdf(0))
        half_width = float(dist.ppf(1 - alpha / 2) - mean_diff)
        ci_lower = mean_diff - half_width
        ci_upper = mean_diff + half_width
    else:
        norm = NormalDist(mean_diff, stderr)
        prob = 1 - norm.cdf(0)
        z = NormalDist().inv_cdf(1 - alpha / 2)
        ci_lower = mean_diff - z * stderr
        ci_upper = mean_diff + z * stderr

    return {
        "baseline_mean": mean_base,
        "variant_mean": mean_var,
        "mean_difference": mean_diff,
        "std_error": stderr,
        "degrees_of_freedom": df,
        "prob_variant_gt_baseline": prob,
        "credible_interval": [ci_lower, ci_upper],
    }


class ScoreSignificanceBaselinePlugin:
    """Compare baseline and variant using effect sizes and t-tests."""

    name = "score_significance"
    config_schema: ClassVar[dict[str, Any]] = _SIGNIFICANCE_SCHEMA

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with treatment and baseline score arrays"
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with p-value and statistical significance fields"
    }

    def __init__(
        self,
        *,
        criteria: Sequence[str] | None = None,
        min_samples: int = 2,
        equal_var: bool = False,
        adjustment: str = "none",
        family_size: int | None = None,
        on_error: str = "abort",
    ) -> None:
        self._criteria = set(criteria) if criteria else None
        self._min_samples = max(int(min_samples), 2)
        self._equal_var = bool(equal_var)
        adjustment = (adjustment or "none").lower()
        if adjustment not in {"none", "bonferroni", "fdr"}:
            adjustment = "none"
        self._adjustment = adjustment
        self._family_size = int(family_size) if family_size else None
        if on_error not in {"abort", "skip"}:
            raise ValueError("on_error must be 'abort' or 'skip'")
        self._on_error = on_error

    def compare(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._compare_impl(baseline, variant)
        except Exception as exc:  # pragma: no cover - defensive guard
            if self._on_error == "skip":
                logger.warning("score_significance skipped due to error: %s", exc)
                return {}
            raise

    def _compare_impl(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        base_scores = collect_scores_by_criterion(baseline)
        var_scores = collect_scores_by_criterion(variant)
        criteria = sorted(set(base_scores.keys()) & set(var_scores.keys()))
        if self._criteria is not None:
            criteria = [name for name in criteria if name in self._criteria]
        raw_p_values: list[tuple[str, float]] = []
        for name in criteria:
            base = base_scores.get(name, [])
            var = var_scores.get(name, [])
            if len(base) < self._min_samples or len(var) < self._min_samples:
                continue
            stats = _compute_significance(base, var, equal_var=self._equal_var)
            if stats:
                results[name] = stats
                p_value = stats.get("p_value")
                if isinstance(p_value, (float, int)):
                    raw_p_values.append((name, float(p_value)))

        if self._adjustment != "none" and raw_p_values:
            family_size = self._family_size or len(raw_p_values)
            if self._adjustment == "bonferroni":
                for name, p_value in raw_p_values:
                    adjusted = min(p_value * family_size, 1.0)
                    result = results.get(name)
                    if result is not None:
                        result["adjusted_p_value"] = adjusted
                        result["adjustment"] = "bonferroni"
            elif self._adjustment == "fdr":
                try:
                    from statsmodels.stats.multitest import fdrcorrection  # type: ignore

                    p_vals = [p for _, p in raw_p_values]
                    _, adj = fdrcorrection(p_vals, alpha=0.05)
                except Exception:
                    adj = benjamini_hochberg([p for _, p in raw_p_values])
                for (name, _), adjusted in zip(raw_p_values, adj):
                    result = results.get(name)
                    if result is not None:
                        result["adjusted_p_value"] = float(adjusted)
                        result["adjustment"] = "fdr"
        return results


class ScoreBayesianBaselinePlugin:
    """Estimate posterior probability that a variant beats the baseline."""

    name = "score_bayes"
    config_schema: ClassVar[dict[str, Any]] = _BAYESIAN_SCHEMA

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with treatment and baseline score arrays"
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row with Bayesian credible interval and probability"
    }

    def __init__(
        self,
        *,
        criteria: Sequence[str] | None = None,
        min_samples: int = 2,
        credible_interval: float = 0.95,
        on_error: str = "abort",
    ) -> None:
        self._criteria = set(criteria) if criteria else None
        self._min_samples = max(int(min_samples), 2)
        self._ci = min(max(float(credible_interval), 0.5), 0.999)
        if on_error not in {"abort", "skip"}:
            raise ValueError("on_error must be 'abort' or 'skip'")
        self._on_error = on_error

    def compare(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._compare_impl(baseline, variant)
        except Exception as exc:  # pragma: no cover - defensive guard
            if self._on_error == "skip":
                logger.warning("score_bayes skipped due to error: %s", exc)
                return {}
            raise

    def _compare_impl(self, baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        base_scores = collect_scores_by_criterion(baseline)
        var_scores = collect_scores_by_criterion(variant)
        criteria = sorted(set(base_scores.keys()) & set(var_scores.keys()))
        if self._criteria is not None:
            criteria = [name for name in criteria if name in self._criteria]
        alpha = 1 - self._ci
        for name in criteria:
            base = base_scores.get(name, [])
            var = var_scores.get(name, [])
            if len(base) < self._min_samples or len(var) < self._min_samples:
                continue
            summary = _compute_bayesian_summary(base, var, alpha)
            if summary:
                results[name] = summary
        return results


register_comparison_plugin(
    "score_significance",
    lambda options: ScoreSignificanceBaselinePlugin(
        criteria=options.get("criteria"),
        min_samples=int(options.get("min_samples", 2)),
        equal_var=bool(options.get("equal_var", False)),
        adjustment=options.get("adjustment", "none"),
        family_size=options.get("family_size"),
        on_error=options.get("on_error", "abort"),
    ),
    schema=_SIGNIFICANCE_SCHEMA,
)

register_comparison_plugin(
    "score_bayes",
    lambda options: ScoreBayesianBaselinePlugin(
        criteria=options.get("criteria"),
        min_samples=int(options.get("min_samples", 2)),
        credible_interval=float(options.get("credible_interval", 0.95)),
        on_error=options.get("on_error", "abort"),
    ),
    schema=_BAYESIAN_SCHEMA,
)
