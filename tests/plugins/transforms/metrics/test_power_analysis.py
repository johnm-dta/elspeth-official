"""Tests for ScorePowerAnalyzer."""

import pytest

from elspeth.plugins.transforms.metrics.power_analysis import ScorePowerAnalyzer


def make_collection(source_field: str, values: list[float]) -> dict:
    """Create a collection dict."""
    return {source_field: values}


class TestScorePowerAnalyzerInit:
    """Tests for plugin initialization."""

    def test_init_with_required_params(self):
        """Initialize with required parameters."""
        analyzer = ScorePowerAnalyzer(input_key="scores")

        assert analyzer.input_key == "scores"
        assert analyzer.source_field == "score"
        assert analyzer._min_samples == 2
        assert analyzer._alpha == 0.05
        assert analyzer._target_power == 0.8
        assert analyzer._effect_size is None
        assert analyzer._null_mean == 0.0
        assert analyzer._on_error == "abort"

    def test_init_with_all_params(self):
        """Initialize with all parameters."""
        analyzer = ScorePowerAnalyzer(
            input_key="my_scores",
            source_field="quality",
            min_samples=10,
            alpha=0.01,
            target_power=0.9,
            effect_size=0.5,
            null_mean=0.5,
            on_error="skip",
        )

        assert analyzer.input_key == "my_scores"
        assert analyzer.source_field == "quality"
        assert analyzer._min_samples == 10
        assert analyzer._alpha == 0.01
        assert analyzer._target_power == 0.9
        assert analyzer._effect_size == 0.5
        assert analyzer._null_mean == 0.5
        assert analyzer._on_error == "skip"

    def test_init_invalid_on_error(self):
        """Invalid on_error raises ValueError."""
        with pytest.raises(ValueError, match="on_error must be"):
            ScorePowerAnalyzer(input_key="scores", on_error="invalid")

    def test_init_min_samples_minimum(self):
        """Min samples has minimum of 2."""
        analyzer = ScorePowerAnalyzer(input_key="scores", min_samples=1)
        assert analyzer._min_samples == 2

        analyzer = ScorePowerAnalyzer(input_key="scores", min_samples=0)
        assert analyzer._min_samples == 2

    def test_init_alpha_bounds(self):
        """Alpha is clamped to valid range."""
        analyzer = ScorePowerAnalyzer(input_key="scores", alpha=0.0)
        assert analyzer._alpha >= 1e-6  # Minimum bound

        analyzer = ScorePowerAnalyzer(input_key="scores", alpha=0.5)
        assert analyzer._alpha <= 0.25  # Maximum bound

    def test_init_target_power_bounds(self):
        """Target power is clamped to valid range."""
        analyzer = ScorePowerAnalyzer(input_key="scores", target_power=0.0)
        assert analyzer._target_power >= 0.1  # Minimum bound

        analyzer = ScorePowerAnalyzer(input_key="scores", target_power=1.0)
        assert analyzer._target_power <= 0.999  # Maximum bound

    def test_stores_config(self):
        """Stores config dict."""
        analyzer = ScorePowerAnalyzer(
            input_key="scores",
            source_field="quality",
            alpha=0.01,
            target_power=0.9,
        )

        assert analyzer.config["input_key"] == "scores"
        assert analyzer.config["source_field"] == "quality"
        assert analyzer.config["alpha"] == 0.01
        assert analyzer.config["target_power"] == 0.9


class TestScorePowerAnalyzerAggregate:
    """Tests for the aggregate method."""

    def test_aggregate_returns_basic_stats(self):
        """Aggregate returns basic statistics."""
        analyzer = ScorePowerAnalyzer(input_key="scores", source_field="score")

        collection = make_collection("score", [1.0, 2.0, 3.0, 4.0, 5.0])
        aggregates = {"scores": collection}

        result = analyzer.aggregate([], aggregates)

        assert "samples" in result
        assert "mean" in result
        assert "std" in result
        assert result["samples"] == 5
        assert result["mean"] == 3.0
        assert result["alpha"] == 0.05
        assert result["target_power"] == 0.8

    def test_aggregate_computes_observed_effect_size(self):
        """Aggregate computes observed effect size from data."""
        analyzer = ScorePowerAnalyzer(
            input_key="scores",
            source_field="score",
            null_mean=0.0,
        )

        # Mean=3.0, std~1.58, effect = (3.0 - 0.0) / 1.58 ~ 1.9
        collection = make_collection("score", [1.0, 2.0, 3.0, 4.0, 5.0])
        aggregates = {"scores": collection}

        result = analyzer.aggregate([], aggregates)

        assert result["observed_effect_size"] is not None
        assert result["observed_effect_size"] > 0

    def test_aggregate_uses_specified_effect_size(self):
        """Aggregate uses specified effect size for calculations."""
        analyzer = ScorePowerAnalyzer(
            input_key="scores",
            source_field="score",
            effect_size=0.5,
        )

        collection = make_collection("score", [1.0, 2.0, 3.0, 4.0, 5.0])
        aggregates = {"scores": collection}

        result = analyzer.aggregate([], aggregates)

        assert result["target_effect_size"] == 0.5

    def test_aggregate_insufficient_samples_returns_empty(self):
        """Returns empty dict when samples < min_samples."""
        analyzer = ScorePowerAnalyzer(
            input_key="scores",
            source_field="score",
            min_samples=10,
        )

        collection = make_collection("score", [1.0, 2.0, 3.0])  # Only 3 samples
        aggregates = {"scores": collection}

        result = analyzer.aggregate([], aggregates)

        assert result == {}

    def test_aggregate_raises_on_missing_input_key(self):
        """Raises KeyError when input_key not in aggregates."""
        analyzer = ScorePowerAnalyzer(input_key="missing", source_field="score")

        aggregates = {"other_key": {}}

        with pytest.raises(KeyError) as exc_info:
            analyzer.aggregate([], aggregates)

        assert "missing" in str(exc_info.value)

    def test_aggregate_raises_on_missing_source_field(self):
        """Raises ValueError when source_field not in collection."""
        analyzer = ScorePowerAnalyzer(input_key="scores", source_field="missing")

        collection = make_collection("score", [1.0, 2.0, 3.0])
        aggregates = {"scores": collection}

        with pytest.raises(ValueError) as exc_info:
            analyzer.aggregate([], aggregates)

        assert "missing" in str(exc_info.value)

    def test_aggregate_power_analysis_without_statsmodels(self):
        """Power analysis returns None values without statsmodels."""
        analyzer = ScorePowerAnalyzer(
            input_key="scores",
            source_field="score",
            effect_size=0.5,
        )

        collection = make_collection("score", [1.0, 2.0, 3.0, 4.0, 5.0])
        aggregates = {"scores": collection}

        result = analyzer.aggregate([], aggregates)

        # Basic stats should always be present
        assert result["samples"] == 5
        assert result["mean"] == 3.0
        assert "std" in result
        assert "observed_effect_size" in result
        # Power analysis may be None without statsmodels
        # (these are optional stats that require statsmodels)

    def test_aggregate_single_sample_std(self):
        """Single sample gives std=0."""
        analyzer = ScorePowerAnalyzer(input_key="scores", source_field="score")

        # Need at least 2 samples for min_samples default
        collection = make_collection("score", [5.0, 5.0])
        aggregates = {"scores": collection}

        result = analyzer.aggregate([], aggregates)

        assert result["std"] == 0.0  # All values identical

    def test_aggregate_zero_std_no_effect_size(self):
        """Zero std gives no observed effect size."""
        analyzer = ScorePowerAnalyzer(input_key="scores", source_field="score")

        collection = make_collection("score", [5.0, 5.0, 5.0])  # All identical
        aggregates = {"scores": collection}

        result = analyzer.aggregate([], aggregates)

        # With zero std, observed_effect_size should be None or not computed
        # since division by zero would occur
        assert result["std"] == 0.0
        assert result["observed_effect_size"] is None


class TestScorePowerAnalyzerErrorHandling:
    """Tests for error handling."""

    def test_on_error_skip_returns_empty_on_missing_key(self):
        """on_error='skip' returns empty dict on missing key."""
        analyzer = ScorePowerAnalyzer(input_key="missing", on_error="skip")

        result = analyzer.aggregate([], {"other": {}})

        assert result == {}

    def test_on_error_skip_returns_empty_on_missing_field(self):
        """on_error='skip' returns empty dict on missing field."""
        analyzer = ScorePowerAnalyzer(
            input_key="scores",
            source_field="missing",
            on_error="skip",
        )

        aggregates = {"scores": {"score": [1.0, 2.0, 3.0]}}

        result = analyzer.aggregate([], aggregates)

        assert result == {}


class TestScorePowerAnalyzerSchema:
    """Tests for schema declarations."""

    def test_has_name(self):
        """Plugin has name attribute."""
        assert ScorePowerAnalyzer.name == "score_power"

    def test_has_config_schema(self):
        """Plugin has config_schema."""
        assert hasattr(ScorePowerAnalyzer, "config_schema")
        schema = ScorePowerAnalyzer.config_schema
        assert schema["type"] == "object"
        assert "input_key" in schema["required"]
        assert "alpha" in schema["properties"]
        assert "target_power" in schema["properties"]
        assert "effect_size" in schema["properties"]

    def test_has_input_schema(self):
        """Plugin has input_schema."""
        assert hasattr(ScorePowerAnalyzer, "input_schema")
        assert ScorePowerAnalyzer.input_schema["type"] == "collection"

    def test_has_output_schema(self):
        """Plugin has output_schema."""
        assert hasattr(ScorePowerAnalyzer, "output_schema")
        assert ScorePowerAnalyzer.output_schema["type"] == "object"


class TestScorePowerAnalyzerRegistration:
    """Tests for plugin registration."""

    def test_plugin_registered(self):
        """Plugin is registered with aggregation factory."""
        from elspeth.core.sda.plugin_registry import create_aggregation_transform

        plugin = create_aggregation_transform({
            "name": "score_power",
            "options": {"input_key": "my_scores", "alpha": 0.01},
        })

        assert isinstance(plugin, ScorePowerAnalyzer)
        assert plugin._alpha == 0.01


class TestScorePowerAnalyzerBackwardCompat:
    """Tests for backward compatibility."""

    def test_aggregator_alias(self):
        """ScorePowerAggregator is alias."""
        from elspeth.plugins.transforms.metrics.power_analysis import (
            ScorePowerAggregator,
        )

        assert ScorePowerAggregator is ScorePowerAnalyzer
