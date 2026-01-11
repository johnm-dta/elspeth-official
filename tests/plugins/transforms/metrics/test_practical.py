"""Tests for ScorePracticalBaselinePlugin."""

import pytest

from elspeth.plugins.transforms.metrics.practical import ScorePracticalBaselinePlugin


def make_payload(scores_by_criterion: dict[str, list[float]]) -> dict:
    """Create a payload with results from criterion scores."""
    results = []
    # Transpose scores_by_criterion into results
    if scores_by_criterion:
        num_records = len(next(iter(scores_by_criterion.values())))
        for i in range(num_records):
            scores = {name: values[i] for name, values in scores_by_criterion.items()}
            results.append({"metrics": {"scores": scores}})
    return {"results": results}


class TestScorePracticalBaselinePluginInit:
    """Tests for plugin initialization."""

    def test_init_with_defaults(self):
        """Initialize with default parameters."""
        plugin = ScorePracticalBaselinePlugin()

        assert plugin._criteria is None
        assert plugin._threshold == 1.0
        assert plugin._success_threshold == 4.0
        assert plugin._min_samples == 1
        assert plugin._on_error == "abort"

    def test_init_with_custom_params(self):
        """Initialize with custom parameters."""
        plugin = ScorePracticalBaselinePlugin(
            criteria=["quality", "accuracy"],
            threshold=0.5,
            success_threshold=3.0,
            min_samples=5,
            on_error="skip",
        )

        assert plugin._criteria == {"quality", "accuracy"}
        assert plugin._threshold == 0.5
        assert plugin._success_threshold == 3.0
        assert plugin._min_samples == 5
        assert plugin._on_error == "skip"

    def test_init_invalid_on_error(self):
        """Invalid on_error raises ValueError."""
        with pytest.raises(ValueError, match="on_error must be"):
            ScorePracticalBaselinePlugin(on_error="invalid")

    def test_init_min_samples_minimum(self):
        """Min samples has minimum of 1."""
        plugin = ScorePracticalBaselinePlugin(min_samples=0)
        assert plugin._min_samples == 1

        plugin = ScorePracticalBaselinePlugin(min_samples=-5)
        assert plugin._min_samples == 1


class TestScorePracticalBaselinePluginCompare:
    """Tests for the compare method."""

    def test_compare_computes_practical_metrics(self):
        """Compare computes all practical significance metrics."""
        plugin = ScorePracticalBaselinePlugin(
            threshold=0.5,
            success_threshold=3.0,
        )

        baseline = make_payload({"quality": [2.5, 3.5, 2.0, 4.0]})
        variant = make_payload({"quality": [3.0, 4.0, 3.5, 4.5]})

        result = plugin.compare(baseline, variant)

        assert "quality" in result
        metrics = result["quality"]
        assert "pairs" in metrics
        assert "mean_difference" in metrics
        assert "median_difference" in metrics
        assert "meaningful_change_rate" in metrics
        assert "success_threshold" in metrics
        assert "baseline_success_rate" in metrics
        assert "variant_success_rate" in metrics
        assert "success_delta" in metrics
        assert "number_needed_to_treat" in metrics

    def test_compare_mean_difference(self):
        """Compare computes correct mean difference."""
        plugin = ScorePracticalBaselinePlugin()

        baseline = make_payload({"quality": [1.0, 2.0, 3.0]})
        variant = make_payload({"quality": [2.0, 3.0, 4.0]})  # All +1

        result = plugin.compare(baseline, variant)

        # Mean diff should be +1.0
        assert abs(result["quality"]["mean_difference"] - 1.0) < 0.001

    def test_compare_median_difference(self):
        """Compare computes correct median difference."""
        plugin = ScorePracticalBaselinePlugin()

        baseline = make_payload({"quality": [1.0, 2.0, 10.0]})  # Outlier
        variant = make_payload({"quality": [2.0, 3.0, 11.0]})  # All +1

        result = plugin.compare(baseline, variant)

        # Median diff should be 1.0 (2->3)
        assert abs(result["quality"]["median_difference"] - 1.0) < 0.001

    def test_compare_meaningful_change_rate(self):
        """Compare computes meaningful change rate correctly."""
        plugin = ScorePracticalBaselinePlugin(threshold=0.5)

        baseline = make_payload({"quality": [1.0, 1.0, 1.0, 1.0]})
        variant = make_payload({"quality": [1.0, 1.6, 1.0, 2.0]})  # 2 meaningful changes

        result = plugin.compare(baseline, variant)

        # 2 out of 4 pairs have |diff| >= 0.5
        assert result["quality"]["meaningful_change_rate"] == 0.5

    def test_compare_success_rates(self):
        """Compare computes success rates correctly."""
        plugin = ScorePracticalBaselinePlugin(success_threshold=3.0)

        baseline = make_payload({"quality": [2.0, 3.0, 4.0, 2.5]})  # 2 successes
        variant = make_payload({"quality": [3.0, 4.0, 4.5, 3.5]})  # 4 successes

        result = plugin.compare(baseline, variant)

        assert result["quality"]["baseline_success_rate"] == 0.5  # 2/4
        assert result["quality"]["variant_success_rate"] == 1.0  # 4/4
        assert result["quality"]["success_delta"] == 0.5  # 1.0 - 0.5

    def test_compare_number_needed_to_treat(self):
        """Compare computes NNT correctly."""
        plugin = ScorePracticalBaselinePlugin(success_threshold=3.0)

        baseline = make_payload({"quality": [2.0, 2.0, 2.0, 2.0]})  # 0 successes
        variant = make_payload({"quality": [3.0, 3.0, 3.0, 3.0]})  # 4 successes

        result = plugin.compare(baseline, variant)

        # Success delta = 1.0, NNT = 1/1.0 = 1
        assert result["quality"]["number_needed_to_treat"] == 1.0

    def test_compare_nnt_infinity_when_no_improvement(self):
        """NNT is infinity when no success improvement."""
        plugin = ScorePracticalBaselinePlugin(success_threshold=3.0)

        baseline = make_payload({"quality": [4.0, 4.0, 4.0]})  # All successes
        variant = make_payload({"quality": [4.0, 4.0, 4.0]})  # All successes

        result = plugin.compare(baseline, variant)

        # No improvement, NNT is infinity
        assert result["quality"]["number_needed_to_treat"] == float("inf")

    def test_compare_filters_by_criteria(self):
        """Compare filters to specified criteria."""
        plugin = ScorePracticalBaselinePlugin(criteria=["quality"])

        baseline = make_payload({"quality": [1.0], "accuracy": [2.0]})
        variant = make_payload({"quality": [2.0], "accuracy": [3.0]})

        result = plugin.compare(baseline, variant)

        assert "quality" in result
        assert "accuracy" not in result

    def test_compare_skips_insufficient_samples(self):
        """Compare skips criteria with insufficient samples."""
        plugin = ScorePracticalBaselinePlugin(min_samples=5)

        baseline = make_payload({"quality": [1.0, 2.0]})  # Only 2 samples
        variant = make_payload({"quality": [2.0, 3.0]})

        result = plugin.compare(baseline, variant)

        assert "quality" not in result

    def test_compare_empty_payloads(self):
        """Compare handles empty payloads."""
        plugin = ScorePracticalBaselinePlugin()

        result = plugin.compare({"results": []}, {"results": []})

        assert result == {}

    def test_compare_mismatched_lengths(self):
        """Compare uses shorter length."""
        plugin = ScorePracticalBaselinePlugin()

        baseline = make_payload({"quality": [1.0, 2.0, 3.0]})
        variant = make_payload({"quality": [2.0]})

        result = plugin.compare(baseline, variant)

        assert result["quality"]["pairs"] == 1


class TestScorePracticalBaselinePluginErrorHandling:
    """Tests for error handling."""

    def test_on_error_skip_returns_empty(self):
        """on_error='skip' returns empty dict on error."""
        plugin = ScorePracticalBaselinePlugin(on_error="skip")

        # Pass malformed data to trigger error in impl
        # The implementation is defensive, so we need to verify it works
        result = plugin.compare({}, {})

        assert result == {}


class TestScorePracticalBaselinePluginSchema:
    """Tests for schema declarations."""

    def test_has_name(self):
        """Plugin has name attribute."""
        assert ScorePracticalBaselinePlugin.name == "score_practical"

    def test_has_config_schema(self):
        """Plugin has config_schema."""
        assert hasattr(ScorePracticalBaselinePlugin, "config_schema")
        schema = ScorePracticalBaselinePlugin.config_schema
        assert schema["type"] == "object"
        assert "threshold" in schema["properties"]
        assert "success_threshold" in schema["properties"]
        assert "min_samples" in schema["properties"]

    def test_has_input_schema(self):
        """Plugin has input_schema."""
        assert hasattr(ScorePracticalBaselinePlugin, "input_schema")
        assert ScorePracticalBaselinePlugin.input_schema["type"] == "object"

    def test_has_output_schema(self):
        """Plugin has output_schema."""
        assert hasattr(ScorePracticalBaselinePlugin, "output_schema")
        assert ScorePracticalBaselinePlugin.output_schema["type"] == "object"


class TestScorePracticalBaselinePluginRegistration:
    """Tests for plugin registration."""

    def test_plugin_registered(self):
        """Plugin is registered with comparison factory."""
        from elspeth.core.sda.plugin_registry import create_baseline_plugin

        plugin = create_baseline_plugin({
            "name": "score_practical",
            "options": {"threshold": 0.5},
        })

        assert isinstance(plugin, ScorePracticalBaselinePlugin)
        assert plugin._threshold == 0.5

    def test_plugin_validates_options(self):
        """Plugin validates options via schema."""
        from elspeth.core.sda.plugin_registry import validate_baseline_plugin_definition

        # Valid definition should not raise
        validate_baseline_plugin_definition({
            "name": "score_practical",
            "options": {
                "threshold": 1.0,
                "success_threshold": 4.0,
            },
        })


class TestScorePracticalBaselinePluginEdgeCases:
    """Edge case tests."""

    def test_zero_threshold(self):
        """Zero threshold means all changes are meaningful (abs(d) >= 0 is always True)."""
        plugin = ScorePracticalBaselinePlugin(threshold=0.0)

        baseline = make_payload({"quality": [1.0, 1.0, 1.0, 1.0]})
        variant = make_payload({"quality": [1.001, 1.0, 1.5, 1.0]})

        result = plugin.compare(baseline, variant)

        # With threshold=0, abs(diff) >= 0 is always True, so all are meaningful
        assert result["quality"]["meaningful_change_rate"] == 1.0

    def test_high_threshold(self):
        """High threshold means few changes are meaningful."""
        plugin = ScorePracticalBaselinePlugin(threshold=100.0)

        baseline = make_payload({"quality": [1.0, 2.0, 3.0]})
        variant = make_payload({"quality": [10.0, 11.0, 12.0]})  # Big changes but < 100

        result = plugin.compare(baseline, variant)

        # No change >= 100, so rate is 0
        assert result["quality"]["meaningful_change_rate"] == 0.0

    def test_negative_differences(self):
        """Handles negative differences (variant worse)."""
        plugin = ScorePracticalBaselinePlugin(threshold=0.5)

        baseline = make_payload({"quality": [5.0, 5.0, 5.0]})
        variant = make_payload({"quality": [4.0, 4.0, 4.0]})  # All -1

        result = plugin.compare(baseline, variant)

        # All diffs are -1, |diff| >= 0.5, so rate is 1.0
        assert result["quality"]["meaningful_change_rate"] == 1.0
        assert result["quality"]["mean_difference"] == -1.0
