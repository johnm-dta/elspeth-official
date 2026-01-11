"""Aggregation analyzer plugins (collection-mode)."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from elspeth.core.sda.plugin_registry import register_aggregation_transform

from .stats_utils import ON_ERROR_SCHEMA, validate_collection_key

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

try:
    import pingouin  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pingouin = None


_STATS_ANALYZER_SCHEMA = {
    "type": "object",
    "required": ["input_key", "source_field"],
    "properties": {
        "input_key": {
            "type": "string",
            "description": "Key to read collection from aggregates dict"
        },
        "source_field": {"type": "string"},
        "flag_field": {"type": "string"},
        "ddof": {"type": "integer", "minimum": 0},
        "on_error": ON_ERROR_SCHEMA,
    },
    "additionalProperties": True,
}

_RECOMMENDATION_SCHEMA = {
    "type": "object",
    "required": ["input_key"],
    "properties": {
        "input_key": {"type": "string", "description": "Collection key in aggregates"},
        "min_samples": {"type": "integer", "minimum": 0},
        "improvement_margin": {"type": "number"},
        "source_field": {"type": "string"},
        "flag_field": {"type": "string"},
        "on_error": ON_ERROR_SCHEMA,
    },
    "additionalProperties": True,
}

_VARIANT_RANKING_SCHEMA = {
    "type": "object",
    "required": ["input_key"],
    "properties": {
        "input_key": {"type": "string", "description": "Collection key in aggregates"},
        "source_field": {"type": "string"},
        "threshold": {"type": "number"},
        "weight_mean": {"type": "number"},
        "weight_pass": {"type": "number"},
    },
    "additionalProperties": True,
}

_AGREEMENT_SCHEMA = {
    "type": "object",
    "required": ["input_key"],
    "properties": {
        "input_key": {"type": "string", "description": "Collection key in aggregates"},
        "criteria": {"type": "array", "items": {"type": "string"}},
        "min_items": {"type": "integer", "minimum": 2},
        "on_error": ON_ERROR_SCHEMA,
    },
    "additionalProperties": True,
}


class ScoreStatsAnalyzer:
    """Analyze score statistics from FieldCollector collection.

    Reads collection from aggregates[input_key] and computes statistics
    (mean, std, min, max, median, count) on specified source_field.

    This analyzer replaces ScoreStatsAggregator. Instead of manually
    collecting rows, it reads from a FieldCollector collection.

    Config options:
        input_key (str, required): Key to read collection from aggregates
        source_field (str, required): Field name containing scores
        flag_field (str, optional): Filter to flagged rows
        ddof (int, optional): Delta degrees of freedom for std (default: 1)

    Example:
        aggregation_plugins:
          - name: field_collector
            output_key: "scores"
          - name: score_stats
            input_key: "scores"
            source_field: "extracted_score"
    """

    name = "score_stats"

    config_schema: ClassVar[dict[str, Any]] = _STATS_ANALYZER_SCHEMA

    # Input schema: collection type (from FieldCollector)
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "collection",
        "description": "Collection with score field arrays",
        "item_schema": {
            "type": "object",
            "properties": {
                "source_field": {"type": "number"}  # Dynamic based on config
            }
        }
    }

    # Output schema: object with statistics
    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Statistical summary of scores",
        "required": ["mean", "std", "count"],
        "properties": {
            "mean": {"type": "number"},
            "std": {"type": "number"},
            "min": {"type": "number"},
            "max": {"type": "number"},
            "median": {"type": "number"},
            "count": {"type": "integer"},
            "flagged_count": {"type": "integer"}
        }
    }

    def __init__(
        self,
        *,
        input_key: str,  # NEW: required parameter
        source_field: str = "score",
        flag_field: str | None = None,
        ddof: int = 1,
        on_error: str = "abort",
    ) -> None:
        """Initialize analyzer.

        Args:
            input_key: Key to read collection from aggregates dict
            source_field: Field name containing scores in collection
            flag_field: Optional field to filter flagged rows
            ddof: Delta degrees of freedom for std calculation
            on_error: Error handling mode ("abort" or "skip")
        """
        self.input_key = input_key
        self.source_field = source_field
        self.flag_field = flag_field
        self.ddof = ddof
        self.on_error = on_error
        # Store config dict for protocol compatibility
        self.config = {
            "input_key": input_key,
            "source_field": source_field,
            "flag_field": flag_field,
            "ddof": ddof,
            "on_error": on_error,
        }

    def aggregate(
        self,
        results: list[dict[str, Any]],
        aggregates: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute statistics from collection.

        Args:
            results: Ignored (kept for protocol compatibility)
            aggregates: Dict containing collection at aggregates[input_key]

        Returns:
            Dict with statistics (mean, std, min, max, median, count)

        Raises:
            KeyError: If input_key not found in aggregates
            ValueError: If source_field not in collection
        """
        # Read collection from aggregates
        collection = validate_collection_key(self.input_key, aggregates)

        # Validate collection has source_field
        if self.source_field not in collection:
            raise ValueError(
                f"Field '{self.source_field}' not found in collection. "
                f"Available fields: {list(collection.keys())}"
            )

        # Extract scores array
        scores = collection[self.source_field]

        # Filter by flag_field if specified
        if self.flag_field and self.flag_field in collection:
            flags = collection[self.flag_field]
            scores = [s for s, f in zip(scores, flags) if f]
            flagged_count = len(scores)
        else:
            flagged_count = None

        # Filter out None and NaN values (FieldCollector uses None for missing values)
        total_count = len(scores)
        valid_scores = []
        for s in scores:
            if s is not None:
                try:
                    num = float(s)
                    if not np.isnan(num):
                        valid_scores.append(num)
                except (TypeError, ValueError):
                    pass  # Skip non-numeric values

        missing_count = total_count - len(valid_scores)

        # Handle empty data
        if not valid_scores:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
                "count": 0,
                "missing_count": missing_count,
                "flagged_count": flagged_count,
            }

        # Compute statistics using numpy
        scores_array = np.array(valid_scores, dtype=float)

        # Guard against ddof >= sample size (would produce NaN)
        std_value = 0.0 if len(valid_scores) <= self.ddof else float(np.std(scores_array, ddof=self.ddof))

        stats = {
            "mean": float(np.mean(scores_array)),
            "std": std_value,
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "median": float(np.median(scores_array)),
            "count": len(valid_scores),
        }

        if missing_count > 0:
            stats["missing_count"] = missing_count

        if flagged_count is not None:
            stats["flagged_count"] = flagged_count

        return stats


# Keep backward compatibility alias (deprecated)
ScoreStatsAggregator = ScoreStatsAnalyzer


class ScoreRecommendationAnalyzer:
    """Recommend next steps based on score analysis.

    Analyzes score statistics from FieldCollector collection and provides
    recommendations (continue, stop, inconclusive) based on sample size and effect size.
    """

    name = "score_recommendation"
    config_schema: ClassVar[dict[str, Any]] = _RECOMMENDATION_SCHEMA

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "collection",
        "description": "Collection with score data"
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Recommendation with justification",
        "required": ["recommendation"],
        "properties": {
            "recommendation": {"type": "string"},
            "reason": {"type": "string"},
            "sample_size": {"type": "integer"},
        }
    }

    def __init__(
        self,
        *,
        input_key: str,
        min_samples: int = 5,
        improvement_margin: float = 0.05,
        source_field: str = "score",
        flag_field: str | None = None,
        on_error: str = "abort",
    ) -> None:
        """Initialize analyzer.

        Args:
            input_key: Key to read collection from aggregates dict
            min_samples: Minimum sample size for confident recommendation
            improvement_margin: Margin above/below 0.5 to consider significant
            source_field: Field name containing scores in collection
            flag_field: Optional field to filter flagged rows
            on_error: Error handling mode ("abort" or "skip")
        """
        self.input_key = input_key
        self._min_samples = min_samples
        self._improvement_margin = improvement_margin
        self.source_field = source_field
        self.flag_field = flag_field
        if on_error not in {"abort", "skip"}:
            raise ValueError("on_error must be 'abort' or 'skip'")
        self._on_error = on_error
        self.config = {
            "input_key": input_key,
            "min_samples": min_samples,
            "improvement_margin": improvement_margin,
            "source_field": source_field,
            "flag_field": flag_field,
            "on_error": on_error,
        }

    def aggregate(
        self,
        results: list[dict[str, Any]],
        aggregates: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate recommendation from collection statistics.

        Args:
            results: Ignored (kept for protocol compatibility)
            aggregates: Dict containing collection at aggregates[input_key]

        Returns:
            Dict with recommendation, reason, and sample_size

        Raises:
            KeyError: If input_key not found in aggregates
            ValueError: If source_field not in collection
        """
        # Read collection from aggregates
        collection = validate_collection_key(self.input_key, aggregates)

        # Validate collection has source_field
        if self.source_field not in collection:
            raise ValueError(
                f"Field '{self.source_field}' not found in collection. "
                f"Available fields: {list(collection.keys())}"
            )

        # Extract scores array
        scores = collection[self.source_field]

        # Filter by flag_field if specified
        if self.flag_field and self.flag_field in collection:
            flags = collection[self.flag_field]
            scores = [s for s, f in zip(scores, flags) if f]

        # Filter out None and NaN values (FieldCollector uses None for missing values)
        valid_scores = []
        for s in scores:
            if s is not None:
                try:
                    num = float(s)
                    if not math.isnan(num):
                        valid_scores.append(num)
                except (TypeError, ValueError):
                    pass

        # Generate recommendation based on sample size and mean
        sample_size = len(valid_scores)

        if sample_size < self._min_samples:
            return {
                "recommendation": "continue",
                "reason": f"Insufficient samples ({sample_size} < {self._min_samples})",
                "sample_size": sample_size,
            }

        mean_score = float(np.mean(valid_scores))

        if mean_score >= (0.5 + self._improvement_margin):
            return {
                "recommendation": "stop",
                "reason": f"Strong performance (mean={mean_score:.3f})",
                "sample_size": sample_size,
                "effect_size": mean_score - 0.5,
            }
        elif mean_score <= (0.5 - self._improvement_margin):
            return {
                "recommendation": "stop",
                "reason": f"Poor performance (mean={mean_score:.3f})",
                "sample_size": sample_size,
                "effect_size": 0.5 - mean_score,
            }
        else:
            return {
                "recommendation": "inconclusive",
                "reason": f"Effect too small (mean={mean_score:.3f})",
                "sample_size": sample_size,
                "effect_size": abs(mean_score - 0.5),
            }


# Backward compatibility alias
ScoreRecommendationAggregator = ScoreRecommendationAnalyzer


class ScoreVariantRankingAnalyzer:
    """Compute composite ranking score from FieldCollector collection."""

    name = "score_variant_ranking"
    config_schema: ClassVar[dict[str, Any]] = _VARIANT_RANKING_SCHEMA

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "collection",
        "description": "Collection with score data"
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Composite ranking with statistics",
        "properties": {
            "samples": {"type": "integer"},
            "mean": {"type": "number"},
            "median": {"type": "number"},
            "min": {"type": "number"},
            "max": {"type": "number"},
            "pass_rate": {"type": "number"},
            "composite_score": {"type": "number"},
        }
    }

    def __init__(
        self,
        *,
        input_key: str,
        source_field: str = "score",
        threshold: float = 0.7,
        weight_mean: float = 1.0,
        weight_pass: float = 1.0,
    ) -> None:
        """Initialize analyzer.

        Args:
            input_key: Key to read collection from aggregates dict
            source_field: Field name containing scores in collection
            threshold: Threshold for pass/fail determination
            weight_mean: Weight for mean score in composite
            weight_pass: Weight for pass rate in composite
        """
        self.input_key = input_key
        self.source_field = source_field
        self._threshold = float(threshold)
        self._weight_mean = float(weight_mean)
        self._weight_pass = float(weight_pass)
        self.config = {
            "input_key": input_key,
            "source_field": source_field,
            "threshold": threshold,
            "weight_mean": weight_mean,
            "weight_pass": weight_pass,
        }

    def aggregate(
        self,
        results: list[dict[str, Any]],
        aggregates: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute ranking from collection.

        Args:
            results: Ignored (kept for protocol compatibility)
            aggregates: Dict containing collection at aggregates[input_key]

        Returns:
            Dict with ranking statistics

        Raises:
            KeyError: If input_key not found in aggregates
            ValueError: If source_field not in collection
        """
        # Read collection from aggregates
        collection = validate_collection_key(self.input_key, aggregates)

        # Validate collection has source_field
        if self.source_field not in collection:
            raise ValueError(
                f"Field '{self.source_field}' not found in collection. "
                f"Available fields: {list(collection.keys())}"
            )

        # Extract scores array and filter None/NaN values
        scores = collection[self.source_field]
        values = []
        for s in scores:
            if s is not None:
                try:
                    num = float(s)
                    if not math.isnan(num):
                        values.append(num)
                except (TypeError, ValueError):
                    pass

        if not values:
            return {}

        # Compute statistics
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        median = float(np.median(arr))
        pass_count = sum(1 for v in values if v >= self._threshold)
        pass_rate = pass_count / len(values)

        # Compute composite score
        composite_score = self._weight_mean * mean + self._weight_pass * pass_rate

        return {
            "samples": len(values),
            "mean": mean,
            "median": median,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "threshold": self._threshold,
            "pass_rate": pass_rate,
            "composite_score": composite_score,
        }


# Backward compatibility alias
ScoreVariantRankingAggregator = ScoreVariantRankingAnalyzer


class ScoreAgreementAnalyzer:
    """Assess agreement/reliability from FieldCollector collection.

    Reads collection from aggregates and computes inter-rater reliability
    metrics across multiple criteria fields.
    """

    name = "score_agreement"
    config_schema: ClassVar[dict[str, Any]] = _AGREEMENT_SCHEMA

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "collection",
        "description": "Collection with multiple criteria score fields"
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Agreement and reliability metrics",
        "properties": {
            "criteria": {"type": "array", "items": {"type": "string"}},
            "cronbach_alpha": {"type": ["number", "null"]},
            "average_correlation": {"type": ["number", "null"]},
            "krippendorff_alpha": {"type": ["number", "null"]},
        }
    }

    def __init__(
        self,
        *,
        input_key: str,
        criteria: Sequence[str] | None = None,
        min_items: int = 2,
        on_error: str = "abort",
    ) -> None:
        """Initialize analyzer.

        Args:
            input_key: Key to read collection from aggregates dict
            criteria: Optional list of criteria fields to analyze
            min_items: Minimum items for reliability calculation
            on_error: Error handling mode ("abort" or "skip")
        """
        self.input_key = input_key
        self._criteria = list(criteria) if criteria else None
        self._min_items = max(int(min_items), 2)
        if on_error not in {"abort", "skip"}:
            raise ValueError("on_error must be 'abort' or 'skip'")
        self._on_error = on_error
        self.config = {
            "input_key": input_key,
            "criteria": criteria,
            "min_items": min_items,
            "on_error": on_error,
        }

    def aggregate(
        self,
        results: list[dict[str, Any]],
        aggregates: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute agreement metrics from collection.

        Args:
            results: Ignored (kept for protocol compatibility)
            aggregates: Dict containing collection at aggregates[input_key]

        Returns:
            Dict with agreement metrics

        Raises:
            KeyError: If input_key not found in aggregates
        """
        try:
            return self._aggregate_impl(aggregates)
        except Exception as exc:  # pragma: no cover - defensive
            if self._on_error == "skip":
                logger.warning("score_agreement skipped due to error: %s", exc)
                return {}
            raise

    def _aggregate_impl(self, aggregates: dict[str, Any]) -> dict[str, Any]:
        # Read collection from aggregates
        collection = validate_collection_key(self.input_key, aggregates)

        # Build matrix from collection fields
        # Collection has fields like: {"criterion1": [scores...], "criterion2": [scores...]}
        matrix = {}
        for field_name, values in collection.items():
            if self._criteria and field_name not in self._criteria:
                continue
            # Filter out None and NaN values
            valid_values = []
            for v in values:
                if v is not None:
                    try:
                        num = float(v)
                        if not math.isnan(num):
                            valid_values.append(num)
                    except (TypeError, ValueError):
                        pass
            if len(valid_values) >= self._min_items:
                matrix[field_name] = valid_values

        usable = {name: values for name, values in matrix.items() if len(values) >= self._min_items}
        if len(usable) < 2:
            return {}

        columns = sorted(usable.keys())
        lengths = [len(usable[name]) for name in columns]
        max_len = max(lengths)
        data = []
        for idx in range(max_len):
            row = []
            for name in columns:
                values = usable[name]
                row.append(values[idx] if idx < len(values) else np.nan)
            data.append(row)
        arr = np.array(data, dtype=float)

        mask = ~np.isnan(arr).all(axis=1)
        arr = arr[mask]
        if arr.shape[0] < self._min_items:
            return {}

        item_variances = np.nanvar(arr, axis=0, ddof=1)
        total_variance = np.nanvar(arr, ddof=1)
        n_items = arr.shape[1]
        if total_variance <= 0 or n_items < 2 or np.isnan(total_variance):
            cronbach_alpha = None
        else:
            cronbach_alpha = (n_items / (n_items - 1)) * (1 - np.nansum(item_variances) / total_variance)

        correlations = []
        for i in range(n_items):
            for j in range(i + 1, n_items):
                col_i = arr[:, i]
                col_j = arr[:, j]
                valid = ~np.isnan(col_i) & ~np.isnan(col_j)
                if valid.sum() >= 2:
                    corr = np.corrcoef(col_i[valid], col_j[valid])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        avg_correlation = float(np.mean(correlations)) if correlations else None

        krippendorff_alpha = None
        if pingouin is not None and arr.shape[1] >= 2:
            try:
                import pandas as pd

                df = pd.DataFrame({columns[i]: arr[:, i] for i in range(n_items)})
                krippendorff_alpha = float(pingouin.krippendorff_alpha(df, reliability_data=True))
            except Exception:  # pragma: no cover - pingouin failure
                krippendorff_alpha = None

        return {
            "criteria": columns,
            "cronbach_alpha": cronbach_alpha,
            "average_correlation": avg_correlation,
            "krippendorff_alpha": krippendorff_alpha,
        }


# Backward compatibility alias
ScoreAgreementAggregator = ScoreAgreementAnalyzer


register_aggregation_transform(
    "score_stats",
    lambda options: ScoreStatsAnalyzer(
        input_key=options["input_key"],  # Required parameter
        source_field=options.get("source_field", "score"),
        flag_field=options.get("flag_field"),
        ddof=int(options.get("ddof", 1)),
        on_error=options.get("on_error", "abort"),
    ),
    schema=_STATS_ANALYZER_SCHEMA,
)

register_aggregation_transform(
    "score_recommendation",
    lambda options: ScoreRecommendationAnalyzer(
        input_key=options["input_key"],
        min_samples=int(options.get("min_samples", 5)),
        improvement_margin=float(options.get("improvement_margin", 0.05)),
        source_field=options.get("source_field", "score"),
        flag_field=options.get("flag_field"),
        on_error=options.get("on_error", "abort"),
    ),
    schema=_RECOMMENDATION_SCHEMA,
)

register_aggregation_transform(
    "score_variant_ranking",
    lambda options: ScoreVariantRankingAnalyzer(
        input_key=options["input_key"],
        source_field=options.get("source_field", "score"),
        threshold=float(options.get("threshold", 0.7)),
        weight_mean=float(options.get("weight_mean", 1.0)),
        weight_pass=float(options.get("weight_pass", 1.0)),
    ),
    schema=_VARIANT_RANKING_SCHEMA,
)

register_aggregation_transform(
    "score_agreement",
    lambda options: ScoreAgreementAnalyzer(
        input_key=options["input_key"],
        criteria=options.get("criteria"),
        min_items=int(options.get("min_items", 2)),
        on_error=options.get("on_error", "abort"),
    ),
    schema=_AGREEMENT_SCHEMA,
)
