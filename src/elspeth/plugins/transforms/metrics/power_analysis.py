"""Statistical power analyzer plugins (collection-mode)."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from elspeth.core.sda.plugin_registry import register_aggregation_transform

from .stats_utils import ON_ERROR_SCHEMA, validate_collection_key

logger = logging.getLogger(__name__)

try:
    from statsmodels.stats.power import TTestPower  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TTestPower = None


_POWER_SCHEMA = {
    "type": "object",
    "required": ["input_key"],
    "properties": {
        "input_key": {"type": "string", "description": "Collection key in aggregates"},
        "source_field": {"type": "string"},
        "criteria": {"type": "array", "items": {"type": "string"}},
        "min_samples": {"type": "integer", "minimum": 2},
        "alpha": {"type": "number", "minimum": 0.0, "maximum": 0.5},
        "target_power": {"type": "number", "minimum": 0.1, "maximum": 0.999},
        "effect_size": {"type": "number", "minimum": 0.0},
        "null_mean": {"type": "number"},
        "on_error": ON_ERROR_SCHEMA,
    },
    "additionalProperties": True,
}


class ScorePowerAnalyzer:
    """Estimate power and required sample size from FieldCollector collection."""

    name = "score_power"
    config_schema: ClassVar[dict[str, Any]] = _POWER_SCHEMA

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "collection",
        "description": "Collection with score data"
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Power analysis results",
        "properties": {
            "samples": {"type": "integer"},
            "mean": {"type": "number"},
            "std": {"type": "number"},
            "observed_effect_size": {"type": ["number", "null"]},
            "required_samples": {"type": ["number", "null"]},
            "achieved_power": {"type": ["number", "null"]},
        }
    }

    def __init__(
        self,
        *,
        input_key: str,
        source_field: str = "score",
        min_samples: int = 2,
        alpha: float = 0.05,
        target_power: float = 0.8,
        effect_size: float | None = None,
        null_mean: float = 0.0,
        on_error: str = "abort",
    ) -> None:
        """Initialize analyzer.

        Args:
            input_key: Key to read collection from aggregates dict
            source_field: Field name containing scores in collection
            min_samples: Minimum sample size for analysis
            alpha: Significance level
            target_power: Target statistical power
            effect_size: Target effect size (if None, uses observed)
            null_mean: Null hypothesis mean
            on_error: Error handling mode ("abort" or "skip")
        """
        self.input_key = input_key
        self.source_field = source_field
        self._min_samples = max(int(min_samples), 2)
        self._alpha = min(max(float(alpha), 1e-6), 0.25)
        self._target_power = min(max(float(target_power), 0.1), 0.999)
        self._effect_size = effect_size
        self._null_mean = float(null_mean)
        if on_error not in {"abort", "skip"}:
            raise ValueError("on_error must be 'abort' or 'skip'")
        self._on_error = on_error
        self.config = {
            "input_key": input_key,
            "source_field": source_field,
            "min_samples": min_samples,
            "alpha": alpha,
            "target_power": target_power,
            "effect_size": effect_size,
            "null_mean": null_mean,
            "on_error": on_error,
        }

    def aggregate(
        self,
        results: list[dict[str, Any]],
        aggregates: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute power analysis from collection.

        Args:
            results: Ignored (kept for protocol compatibility)
            aggregates: Dict containing collection at aggregates[input_key]

        Returns:
            Dict with power analysis results

        Raises:
            KeyError: If input_key not found in aggregates
            ValueError: If source_field not in collection
        """
        try:
            return self._aggregate_impl(aggregates)
        except Exception as exc:  # pragma: no cover - defensive
            if self._on_error == "skip":
                logger.warning("score_power skipped due to error: %s", exc)
                return {}
            raise

    def _aggregate_impl(self, aggregates: dict[str, Any]) -> dict[str, Any]:
        # Read collection from aggregates
        collection = validate_collection_key(self.input_key, aggregates)

        # Validate collection has source_field
        if self.source_field not in collection:
            raise ValueError(
                f"Field '{self.source_field}' not found in collection. "
                f"Available fields: {list(collection.keys())}"
            )

        # Extract scores array
        values = collection[self.source_field]

        if len(values) < self._min_samples:
            return {}

        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        n = arr.size

        observed_effect = None
        if std > 0:
            observed_effect = (mean - self._null_mean) / std
        effect = self._effect_size or observed_effect

        required_n = None
        achieved_power = None
        if effect and effect > 0 and TTestPower is not None:
            try:
                test = TTestPower()
                required_n = test.solve_power(
                    effect_size=effect,
                    alpha=self._alpha,
                    power=self._target_power,
                    alternative="two-sided",
                )
                if observed_effect:
                    achieved_power = test.solve_power(
                        effect_size=observed_effect,
                        alpha=self._alpha,
                        nobs=n,
                        alternative="two-sided",
                    )
            except Exception:  # pragma: no cover
                required_n = None
                achieved_power = None

        return {
            "samples": n,
            "mean": mean,
            "std": std,
            "observed_effect_size": observed_effect,
            "target_effect_size": effect,
            "required_samples": float(required_n) if required_n is not None else None,
            "achieved_power": float(achieved_power) if achieved_power is not None else None,
            "alpha": self._alpha,
            "target_power": self._target_power,
        }


# Backward compatibility alias
ScorePowerAggregator = ScorePowerAnalyzer


register_aggregation_transform(
    "score_power",
    lambda options: ScorePowerAnalyzer(
        input_key=options["input_key"],
        source_field=options.get("source_field", "score"),
        min_samples=int(options.get("min_samples", 2)),
        alpha=float(options.get("alpha", 0.05)),
        target_power=float(options.get("target_power", 0.8)),
        effect_size=options.get("effect_size"),
        null_mean=float(options.get("null_mean", 0.0)),
        on_error=options.get("on_error", "abort"),
    ),
    schema=_POWER_SCHEMA,
)
