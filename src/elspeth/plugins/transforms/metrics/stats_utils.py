"""Shared statistical utilities for metrics plugins."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    scipy_stats = None

# Error handling schema
ON_ERROR_SCHEMA = {"type": "string", "enum": ["abort", "skip"]}


def validate_collection_key(
    input_key: str,
    aggregates: dict[str, Any],
) -> Any:
    """Validate and retrieve collection from aggregates.

    Args:
        input_key: Key to look up in aggregates
        aggregates: Aggregates dict from ResultAggregator

    Returns:
        Collection dict

    Raises:
        KeyError: If input_key not found with helpful message
    """
    if input_key not in aggregates:
        raise KeyError(
            f"Collection key '{input_key}' not found in aggregates. "
            f"Available keys: {list(aggregates.keys())}. "
            f"Did you forget to add FieldCollector with output_key='{input_key}'?"
        )

    return aggregates[input_key]


def extract_scores(
    collection: dict[str, Any],
    source_field: str,
    flag_field: str | None = None,
) -> Any:
    """Extract scores from collection, optionally filtering by flags.

    Args:
        collection: Dict with array values (from FieldCollector)
        source_field: Field name containing scores
        flag_field: Optional field to filter by (truthy values kept)

    Returns:
        List of float scores

    Raises:
        ValueError: If source_field not in collection
    """
    if source_field not in collection:
        raise ValueError(
            f"Field '{source_field}' not found in collection. "
            f"Available fields: {list(collection.keys())}"
        )

    scores = collection[source_field]

    # Filter by flags if specified
    if flag_field and flag_field in collection:
        flags = collection[flag_field]
        scores = [s for s, f in zip(scores, flags) if f]

    return scores


def safe_mean(values: list[float]) -> float | None:
    """Compute mean, returning None for empty list."""
    if not values:
        return None
    return float(np.mean(values))


def safe_std(values: list[float], ddof: int = 1) -> float | None:
    """Compute standard deviation, returning None for empty list."""
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=ddof))


def collect_scores_by_criterion(payload: Mapping[str, Any]) -> dict[str, list[float]]:
    """Collect scores by criterion from payload results.

    Args:
        payload: Dict with 'results' key containing list of records

    Returns:
        Dict mapping criterion names to lists of float scores
    """
    scores_by_name: dict[str, list[float]] = {}
    for record in payload.get("results", []) or []:
        metrics = record.get("metrics") or {}
        scores = metrics.get("scores") or {}
        if not isinstance(scores, Mapping):
            continue
        for name, value in scores.items():
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(number):
                continue
            scores_by_name.setdefault(name, []).append(number)
    return scores_by_name


def collect_paired_scores_by_criterion(
    baseline: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, list[tuple[float, float]]]:
    """Collect paired scores by criterion from baseline and variant.

    Args:
        baseline: Baseline payload with 'results' key
        variant: Variant payload with 'results' key

    Returns:
        Dict mapping criterion names to lists of (baseline, variant) score tuples
    """
    baseline_results = baseline.get("results", []) or []
    variant_results = variant.get("results", []) or []
    count = min(len(baseline_results), len(variant_results))
    pairs: dict[str, list[tuple[float, float]]] = {}
    for index in range(count):
        base_metrics = (baseline_results[index].get("metrics") if isinstance(baseline_results[index], Mapping) else {}) or {}
        var_metrics = (variant_results[index].get("metrics") if isinstance(variant_results[index], Mapping) else {}) or {}
        base_scores = base_metrics.get("scores") or {}
        var_scores = var_metrics.get("scores") or {}
        if not isinstance(base_scores, Mapping) or not isinstance(var_scores, Mapping):
            continue
        for name, base_value in base_scores.items():
            if name not in var_scores:
                continue
            try:
                base_number = float(base_value)
                var_number = float(var_scores[name])
            except (TypeError, ValueError):
                continue
            if math.isnan(base_number) or math.isnan(var_number):
                continue
            pairs.setdefault(name, []).append((base_number, var_number))
    return pairs


def calculate_cliffs_delta(group1: Sequence[float], group2: Sequence[float]) -> tuple[float, str]:
    """Calculate Cliff's Delta effect size.

    Args:
        group1: First group of scores
        group2: Second group of scores

    Returns:
        Tuple of (delta, interpretation) where interpretation is
        "negligible", "small", "medium", or "large"
    """
    arr1 = np.asarray(list(group1), dtype=float)
    arr2 = np.asarray(list(group2), dtype=float)
    if arr1.size == 0 or arr2.size == 0:
        return 0.0, "no_data"
    dominance = 0
    for x in arr1:
        dominance += np.sum(arr2 > x)
        dominance -= np.sum(arr2 < x)
    delta = dominance / (arr1.size * arr2.size)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"
    return float(delta), interpretation


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction to p-values.

    Args:
        p_values: List of p-values to correct

    Returns:
        List of corrected p-values
    """
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    prev = 1.0
    for rank, idx in reversed(list(enumerate(sorted_indices, start=1))):
        corrected = p_values[idx] * m / rank
        value = min(corrected, prev)
        adjusted[idx] = min(value, 1.0)
        prev = adjusted[idx]
    return adjusted
