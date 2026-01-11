"""Tests for ScoreDistributionAnalyzer."""

import pytest

from elspeth.plugins.transforms.metrics.distribution import (
    ScoreDistributionAnalyzer,
    _compute_distribution_shift,
)


def make_payload(scores_by_criterion: dict[str, list[float]]) -> dict:
    """Create a payload with results from criterion scores."""
    results = []
    if scores_by_criterion:
        num_records = len(next(iter(scores_by_criterion.values())))
        for i in range(num_records):
            scores = {name: values[i] for name, values in scores_by_criterion.items()}
            results.append({"metrics": {"scores": scores}})
    return {"results": results}


# =============================================================================
# Tests for _compute_distribution_shift helper
# =============================================================================


class TestComputeDistributionShift:
    """Tests for the _compute_distribution_shift helper function."""

    def test_returns_all_metrics(self):
        """Returns dict with all expected keys."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        variant = [2.0, 3.0, 4.0, 5.0, 6.0]

        result = _compute_distribution_shift(baseline, variant)

        assert "baseline_samples" in result
        assert "variant_samples" in result
        assert "baseline_mean" in result
        assert "variant_mean" in result
        assert "baseline_std" in result
        assert "variant_std" in result
        assert "ks_statistic" in result
        assert "ks_pvalue" in result
        assert "mwu_statistic" in result
        assert "mwu_pvalue" in result
        assert "js_divergence" in result

    def test_computes_sample_sizes(self):
        """Computes sample sizes correctly."""
        baseline = [1.0, 2.0, 3.0]
        variant = [4.0, 5.0]

        result = _compute_distribution_shift(baseline, variant)

        assert result["baseline_samples"] == 3
        assert result["variant_samples"] == 2

    def test_computes_means_and_stds(self):
        """Computes means and stds correctly."""
        baseline = [1.0, 2.0, 3.0]
        variant = [4.0, 5.0, 6.0]

        result = _compute_distribution_shift(baseline, variant)

        assert result["baseline_mean"] == 2.0
        assert result["variant_mean"] == 5.0
        assert result["baseline_std"] > 0
        assert result["variant_std"] > 0

    def test_ks_statistic_identical_distributions(self):
        """KS statistic is small for identical distributions (requires scipy)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = _compute_distribution_shift(data, data)

        # KS test requires scipy - may be None if not available
        if result["ks_statistic"] is not None:
            assert result["ks_statistic"] == 0.0
            assert result["ks_pvalue"] is not None

    def test_ks_statistic_different_distributions(self):
        """KS statistic is large for different distributions (requires scipy)."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        variant = [10.0, 11.0, 12.0, 13.0, 14.0]  # No overlap

        result = _compute_distribution_shift(baseline, variant)

        # KS test requires scipy - may be None if not available
        if result["ks_statistic"] is not None:
            assert result["ks_statistic"] > 0.5  # Large difference

    def test_mann_whitney_u_statistic(self):
        """Mann-Whitney U test is computed (requires scipy)."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        variant = [2.0, 3.0, 4.0, 5.0, 6.0]

        result = _compute_distribution_shift(baseline, variant)

        # Mann-Whitney requires scipy - may be None if not available
        if result["mwu_statistic"] is not None:
            assert result["mwu_pvalue"] is not None

    def test_jensen_shannon_divergence(self):
        """JS divergence is computed."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        variant = [2.0, 3.0, 4.0, 5.0, 6.0]

        result = _compute_distribution_shift(baseline, variant)

        assert result["js_divergence"] is not None
        assert result["js_divergence"] >= 0

    def test_js_divergence_zero_for_identical_ranges(self):
        """JS divergence is zero when all values identical."""
        data = [5.0, 5.0, 5.0, 5.0, 5.0]

        result = _compute_distribution_shift(data, data)

        # When range is zero, divergence should be 0.0
        assert result["js_divergence"] == 0.0


# =============================================================================
# Tests for ScoreDistributionAnalyzer
# =============================================================================


class TestScoreDistributionAnalyzerInit:
    """Tests for plugin initialization."""

    def test_init_with_required_params(self):
        """Initialize with required parameters."""
        analyzer = ScoreDistributionAnalyzer(input_key="scores")

        assert analyzer.input_key == "scores"
        assert analyzer.source_field == "score"
        assert analyzer._criteria is None
        assert analyzer._min_samples == 2
        assert analyzer._on_error == "abort"

    def test_init_with_all_params(self):
        """Initialize with all parameters."""
        analyzer = ScoreDistributionAnalyzer(
            input_key="my_scores",
            source_field="quality",
            criteria=["quality", "accuracy"],
            min_samples=10,
            on_error="skip",
        )

        assert analyzer.input_key == "my_scores"
        assert analyzer.source_field == "quality"
        assert analyzer._criteria == {"quality", "accuracy"}
        assert analyzer._min_samples == 10
        assert analyzer._on_error == "skip"

    def test_init_invalid_on_error(self):
        """Invalid on_error raises ValueError."""
        with pytest.raises(ValueError, match="on_error must be"):
            ScoreDistributionAnalyzer(input_key="scores", on_error="invalid")

    def test_init_min_samples_minimum(self):
        """Min samples has minimum of 2."""
        analyzer = ScoreDistributionAnalyzer(input_key="scores", min_samples=1)
        assert analyzer._min_samples == 2

        analyzer = ScoreDistributionAnalyzer(input_key="scores", min_samples=0)
        assert analyzer._min_samples == 2

    def test_stores_config(self):
        """Stores config dict."""
        analyzer = ScoreDistributionAnalyzer(
            input_key="scores",
            source_field="quality",
            min_samples=5,
        )

        assert analyzer.config["input_key"] == "scores"
        assert analyzer.config["source_field"] == "quality"
        assert analyzer.config["min_samples"] == 5


class TestScoreDistributionAnalyzerAggregate:
    """Tests for the aggregate method."""

    def test_aggregate_returns_empty(self):
        """Aggregate returns empty dict (used as comparison plugin)."""
        analyzer = ScoreDistributionAnalyzer(input_key="scores")

        result = analyzer.aggregate([], {})

        assert result == {}


class TestScoreDistributionAnalyzerCompare:
    """Tests for the compare method."""

    def test_compare_computes_distribution_metrics(self):
        """Compare computes all distribution metrics."""
        analyzer = ScoreDistributionAnalyzer(input_key="scores")

        baseline = make_payload({"quality": [1.0, 2.0, 3.0, 4.0, 5.0]})
        variant = make_payload({"quality": [2.0, 3.0, 4.0, 5.0, 6.0]})

        result = analyzer.compare(baseline, variant)

        assert "quality" in result
        metrics = result["quality"]
        assert "ks_statistic" in metrics
        assert "mwu_statistic" in metrics
        assert "js_divergence" in metrics

    def test_compare_filters_by_criteria(self):
        """Compare filters to specified criteria."""
        analyzer = ScoreDistributionAnalyzer(
            input_key="scores",
            criteria=["quality"],
        )

        baseline = make_payload({"quality": [1.0, 2.0, 3.0], "accuracy": [1.0, 2.0, 3.0]})
        variant = make_payload({"quality": [2.0, 3.0, 4.0], "accuracy": [2.0, 3.0, 4.0]})

        result = analyzer.compare(baseline, variant)

        assert "quality" in result
        assert "accuracy" not in result

    def test_compare_skips_insufficient_samples(self):
        """Compare skips criteria with insufficient samples."""
        analyzer = ScoreDistributionAnalyzer(input_key="scores", min_samples=10)

        baseline = make_payload({"quality": [1.0, 2.0, 3.0]})  # Only 3 samples
        variant = make_payload({"quality": [2.0, 3.0, 4.0]})

        result = analyzer.compare(baseline, variant)

        assert "quality" not in result

    def test_compare_empty_payloads(self):
        """Compare handles empty payloads."""
        analyzer = ScoreDistributionAnalyzer(input_key="scores")

        result = analyzer.compare({"results": []}, {"results": []})

        assert result == {}

    def test_compare_multiple_criteria(self):
        """Compare handles multiple criteria."""
        analyzer = ScoreDistributionAnalyzer(input_key="scores")

        baseline = make_payload({
            "quality": [1.0, 2.0, 3.0, 4.0, 5.0],
            "accuracy": [2.0, 3.0, 4.0, 5.0, 6.0],
        })
        variant = make_payload({
            "quality": [2.0, 3.0, 4.0, 5.0, 6.0],
            "accuracy": [3.0, 4.0, 5.0, 6.0, 7.0],
        })

        result = analyzer.compare(baseline, variant)

        assert "quality" in result
        assert "accuracy" in result


class TestScoreDistributionAnalyzerErrorHandling:
    """Tests for error handling."""

    def test_on_error_skip_returns_empty(self):
        """on_error='skip' returns empty dict on error."""
        analyzer = ScoreDistributionAnalyzer(input_key="scores", on_error="skip")

        # Empty payloads don't raise - behavior is defensive
        result = analyzer.compare({}, {})

        assert result == {}


class TestScoreDistributionAnalyzerSchema:
    """Tests for schema declarations."""

    def test_has_name(self):
        """Plugin has name attribute."""
        assert ScoreDistributionAnalyzer.name == "score_distribution"

    def test_has_config_schema(self):
        """Plugin has config_schema."""
        assert hasattr(ScoreDistributionAnalyzer, "config_schema")
        schema = ScoreDistributionAnalyzer.config_schema
        assert schema["type"] == "object"
        assert "input_key" in schema["properties"]
        assert "source_field" in schema["properties"]
        assert "min_samples" in schema["properties"]

    def test_has_input_schema(self):
        """Plugin has input_schema."""
        assert hasattr(ScoreDistributionAnalyzer, "input_schema")
        assert ScoreDistributionAnalyzer.input_schema["type"] == "collection"

    def test_has_output_schema(self):
        """Plugin has output_schema."""
        assert hasattr(ScoreDistributionAnalyzer, "output_schema")
        assert ScoreDistributionAnalyzer.output_schema["type"] == "object"


class TestScoreDistributionAnalyzerRegistration:
    """Tests for plugin registration."""

    def test_plugin_registered(self):
        """Plugin is registered with baseline factory."""
        from elspeth.core.sda.plugin_registry import create_baseline_plugin

        plugin = create_baseline_plugin({
            "name": "score_distribution",
            "options": {"input_key": "my_scores", "min_samples": 5},
        })

        assert isinstance(plugin, ScoreDistributionAnalyzer)
        assert plugin._min_samples == 5


class TestScoreDistributionAnalyzerBackwardCompat:
    """Tests for backward compatibility."""

    def test_aggregator_alias(self):
        """ScoreDistributionAggregator is alias."""
        from elspeth.plugins.transforms.metrics.distribution import (
            ScoreDistributionAggregator,
        )

        assert ScoreDistributionAggregator is ScoreDistributionAnalyzer
