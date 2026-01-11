"""Tests for ScoreSignificanceBaselinePlugin and ScoreBayesianBaselinePlugin."""

import pytest

from elspeth.plugins.transforms.metrics.significance import (
    ScoreBayesianBaselinePlugin,
    ScoreSignificanceBaselinePlugin,
    _compute_bayesian_summary,
    _compute_significance,
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
# Tests for _compute_significance helper
# =============================================================================


class TestComputeSignificance:
    """Tests for the _compute_significance helper function."""

    def test_returns_all_metrics(self):
        """Returns dict with all expected keys."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        variant = [2.0, 3.0, 4.0, 5.0, 6.0]

        result = _compute_significance(baseline, variant)

        assert "baseline_mean" in result
        assert "variant_mean" in result
        assert "mean_difference" in result
        assert "baseline_std" in result
        assert "variant_std" in result
        assert "baseline_samples" in result
        assert "variant_samples" in result
        assert "effect_size" in result
        assert "t_stat" in result
        assert "degrees_of_freedom" in result
        assert "p_value" in result

    def test_computes_means(self):
        """Computes means correctly."""
        baseline = [1.0, 2.0, 3.0]
        variant = [4.0, 5.0, 6.0]

        result = _compute_significance(baseline, variant)

        assert result["baseline_mean"] == 2.0
        assert result["variant_mean"] == 5.0
        assert result["mean_difference"] == 3.0

    def test_computes_sample_sizes(self):
        """Computes sample sizes correctly."""
        baseline = [1.0, 2.0, 3.0]
        variant = [4.0, 5.0]

        result = _compute_significance(baseline, variant)

        assert result["baseline_samples"] == 3
        assert result["variant_samples"] == 2

    def test_computes_effect_size(self):
        """Computes Cohen's d effect size."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        variant = [6.0, 7.0, 8.0, 9.0, 10.0]  # Clear separation

        result = _compute_significance(baseline, variant)

        # Effect size should be positive and large
        assert result["effect_size"] is not None
        assert result["effect_size"] > 2.0  # Very large effect

    def test_handles_equal_variance(self):
        """Can compute with equal_var=True."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        variant = [2.0, 3.0, 4.0, 5.0, 6.0]

        result = _compute_significance(baseline, variant, equal_var=True)

        # Should still compute all metrics
        assert result["t_stat"] is not None
        assert result["degrees_of_freedom"] == 8.0  # n1 + n2 - 2

    def test_handles_empty_groups(self):
        """Handles empty groups gracefully."""
        result = _compute_significance([], [1.0, 2.0, 3.0])

        assert result["baseline_samples"] == 0
        assert result["baseline_mean"] == 0.0
        # t_stat may still be computed if denominator is non-zero

    def test_handles_single_sample(self):
        """Handles single sample in group."""
        result = _compute_significance([1.0], [2.0, 3.0, 4.0])

        assert result["baseline_samples"] == 1
        assert result["baseline_std"] == 0.0


# =============================================================================
# Tests for _compute_bayesian_summary helper
# =============================================================================


class TestComputeBayesianSummary:
    """Tests for the _compute_bayesian_summary helper function."""

    def test_returns_all_metrics(self):
        """Returns dict with all expected keys."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        variant = [2.0, 3.0, 4.0, 5.0, 6.0]

        result = _compute_bayesian_summary(baseline, variant, alpha=0.05)

        assert "baseline_mean" in result
        assert "variant_mean" in result
        assert "mean_difference" in result
        assert "std_error" in result
        assert "prob_variant_gt_baseline" in result
        assert "credible_interval" in result

    def test_computes_probability(self):
        """Computes probability that variant > baseline."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        variant = [6.0, 7.0, 8.0, 9.0, 10.0]  # Clearly higher

        result = _compute_bayesian_summary(baseline, variant, alpha=0.05)

        # Should be very high probability
        assert result["prob_variant_gt_baseline"] > 0.99

    def test_computes_credible_interval(self):
        """Computes credible interval."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        variant = [2.0, 3.0, 4.0, 5.0, 6.0]

        result = _compute_bayesian_summary(baseline, variant, alpha=0.05)

        ci = result["credible_interval"]
        assert isinstance(ci, list)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower < upper

    def test_handles_zero_stderr(self):
        """Returns empty dict when stderr is zero."""
        baseline = [1.0, 1.0, 1.0]  # Zero variance
        variant = [1.0, 1.0, 1.0]

        result = _compute_bayesian_summary(baseline, variant, alpha=0.05)

        assert result == {}


# =============================================================================
# Tests for ScoreSignificanceBaselinePlugin
# =============================================================================


class TestScoreSignificanceBaselinePluginInit:
    """Tests for plugin initialization."""

    def test_init_with_defaults(self):
        """Initialize with default parameters."""
        plugin = ScoreSignificanceBaselinePlugin()

        assert plugin._criteria is None
        assert plugin._min_samples == 2
        assert plugin._equal_var is False
        assert plugin._adjustment == "none"
        assert plugin._family_size is None
        assert plugin._on_error == "abort"

    def test_init_with_custom_params(self):
        """Initialize with custom parameters."""
        plugin = ScoreSignificanceBaselinePlugin(
            criteria=["quality", "accuracy"],
            min_samples=10,
            equal_var=True,
            adjustment="bonferroni",
            family_size=5,
            on_error="skip",
        )

        assert plugin._criteria == {"quality", "accuracy"}
        assert plugin._min_samples == 10
        assert plugin._equal_var is True
        assert plugin._adjustment == "bonferroni"
        assert plugin._family_size == 5
        assert plugin._on_error == "skip"

    def test_init_invalid_on_error(self):
        """Invalid on_error raises ValueError."""
        with pytest.raises(ValueError, match="on_error must be"):
            ScoreSignificanceBaselinePlugin(on_error="invalid")

    def test_init_min_samples_minimum(self):
        """Min samples has minimum of 2."""
        plugin = ScoreSignificanceBaselinePlugin(min_samples=1)
        assert plugin._min_samples == 2

        plugin = ScoreSignificanceBaselinePlugin(min_samples=0)
        assert plugin._min_samples == 2

    def test_init_invalid_adjustment_defaults_to_none(self):
        """Invalid adjustment defaults to none."""
        plugin = ScoreSignificanceBaselinePlugin(adjustment="invalid")
        assert plugin._adjustment == "none"


class TestScoreSignificanceBaselinePluginCompare:
    """Tests for the compare method."""

    def test_compare_computes_significance_metrics(self):
        """Compare computes all significance metrics."""
        plugin = ScoreSignificanceBaselinePlugin()

        baseline = make_payload({"quality": [1.0, 2.0, 3.0, 4.0, 5.0]})
        variant = make_payload({"quality": [2.0, 3.0, 4.0, 5.0, 6.0]})

        result = plugin.compare(baseline, variant)

        assert "quality" in result
        metrics = result["quality"]
        assert "baseline_mean" in metrics
        assert "variant_mean" in metrics
        assert "t_stat" in metrics
        assert "p_value" in metrics

    def test_compare_filters_by_criteria(self):
        """Compare filters to specified criteria."""
        plugin = ScoreSignificanceBaselinePlugin(criteria=["quality"])

        baseline = make_payload({"quality": [1.0, 2.0, 3.0], "accuracy": [1.0, 2.0, 3.0]})
        variant = make_payload({"quality": [2.0, 3.0, 4.0], "accuracy": [2.0, 3.0, 4.0]})

        result = plugin.compare(baseline, variant)

        assert "quality" in result
        assert "accuracy" not in result

    def test_compare_skips_insufficient_samples(self):
        """Compare skips criteria with insufficient samples."""
        plugin = ScoreSignificanceBaselinePlugin(min_samples=10)

        baseline = make_payload({"quality": [1.0, 2.0, 3.0]})  # Only 3 samples
        variant = make_payload({"quality": [2.0, 3.0, 4.0]})

        result = plugin.compare(baseline, variant)

        assert "quality" not in result

    def test_compare_bonferroni_adjustment(self):
        """Compare applies Bonferroni correction."""
        plugin = ScoreSignificanceBaselinePlugin(adjustment="bonferroni")

        baseline = make_payload({
            "quality": [1.0, 2.0, 3.0, 4.0, 5.0],
            "accuracy": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        variant = make_payload({
            "quality": [2.0, 3.0, 4.0, 5.0, 6.0],
            "accuracy": [2.0, 3.0, 4.0, 5.0, 6.0],
        })

        result = plugin.compare(baseline, variant)

        # Both should have adjusted_p_value
        if result.get("quality", {}).get("p_value") is not None:
            assert "adjusted_p_value" in result["quality"]
            assert result["quality"]["adjustment"] == "bonferroni"

    def test_compare_fdr_adjustment(self):
        """Compare applies FDR correction."""
        plugin = ScoreSignificanceBaselinePlugin(adjustment="fdr")

        baseline = make_payload({
            "quality": [1.0, 2.0, 3.0, 4.0, 5.0],
            "accuracy": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        variant = make_payload({
            "quality": [2.0, 3.0, 4.0, 5.0, 6.0],
            "accuracy": [2.0, 3.0, 4.0, 5.0, 6.0],
        })

        result = plugin.compare(baseline, variant)

        # Both should have adjusted_p_value
        if result.get("quality", {}).get("p_value") is not None:
            assert "adjusted_p_value" in result["quality"]
            assert result["quality"]["adjustment"] == "fdr"

    def test_compare_empty_payloads(self):
        """Compare handles empty payloads."""
        plugin = ScoreSignificanceBaselinePlugin()

        result = plugin.compare({"results": []}, {"results": []})

        assert result == {}


class TestScoreSignificanceBaselinePluginSchema:
    """Tests for schema declarations."""

    def test_has_name(self):
        """Plugin has name attribute."""
        assert ScoreSignificanceBaselinePlugin.name == "score_significance"

    def test_has_config_schema(self):
        """Plugin has config_schema."""
        assert hasattr(ScoreSignificanceBaselinePlugin, "config_schema")
        schema = ScoreSignificanceBaselinePlugin.config_schema
        assert schema["type"] == "object"
        assert "min_samples" in schema["properties"]
        assert "equal_var" in schema["properties"]
        assert "adjustment" in schema["properties"]

    def test_has_input_schema(self):
        """Plugin has input_schema."""
        assert hasattr(ScoreSignificanceBaselinePlugin, "input_schema")
        assert ScoreSignificanceBaselinePlugin.input_schema["type"] == "object"

    def test_has_output_schema(self):
        """Plugin has output_schema."""
        assert hasattr(ScoreSignificanceBaselinePlugin, "output_schema")
        assert ScoreSignificanceBaselinePlugin.output_schema["type"] == "object"


class TestScoreSignificanceBaselinePluginRegistration:
    """Tests for plugin registration."""

    def test_plugin_registered(self):
        """Plugin is registered with baseline factory."""
        from elspeth.core.sda.plugin_registry import create_baseline_plugin

        plugin = create_baseline_plugin({
            "name": "score_significance",
            "options": {"min_samples": 5},
        })

        assert isinstance(plugin, ScoreSignificanceBaselinePlugin)
        assert plugin._min_samples == 5


# =============================================================================
# Tests for ScoreBayesianBaselinePlugin
# =============================================================================


class TestScoreBayesianBaselinePluginInit:
    """Tests for plugin initialization."""

    def test_init_with_defaults(self):
        """Initialize with default parameters."""
        plugin = ScoreBayesianBaselinePlugin()

        assert plugin._criteria is None
        assert plugin._min_samples == 2
        assert plugin._ci == 0.95
        assert plugin._on_error == "abort"

    def test_init_with_custom_params(self):
        """Initialize with custom parameters."""
        plugin = ScoreBayesianBaselinePlugin(
            criteria=["quality"],
            min_samples=10,
            credible_interval=0.99,
            on_error="skip",
        )

        assert plugin._criteria == {"quality"}
        assert plugin._min_samples == 10
        assert plugin._ci == 0.99
        assert plugin._on_error == "skip"

    def test_init_invalid_on_error(self):
        """Invalid on_error raises ValueError."""
        with pytest.raises(ValueError, match="on_error must be"):
            ScoreBayesianBaselinePlugin(on_error="invalid")

    def test_init_credible_interval_bounds(self):
        """Credible interval clamped to valid range."""
        plugin = ScoreBayesianBaselinePlugin(credible_interval=0.1)
        assert plugin._ci == 0.5  # Clamped to minimum

        plugin = ScoreBayesianBaselinePlugin(credible_interval=1.5)
        assert plugin._ci == 0.999  # Clamped to maximum


class TestScoreBayesianBaselinePluginCompare:
    """Tests for the compare method."""

    def test_compare_computes_bayesian_metrics(self):
        """Compare computes all Bayesian metrics."""
        plugin = ScoreBayesianBaselinePlugin()

        baseline = make_payload({"quality": [1.0, 2.0, 3.0, 4.0, 5.0]})
        variant = make_payload({"quality": [2.0, 3.0, 4.0, 5.0, 6.0]})

        result = plugin.compare(baseline, variant)

        assert "quality" in result
        metrics = result["quality"]
        assert "baseline_mean" in metrics
        assert "variant_mean" in metrics
        assert "prob_variant_gt_baseline" in metrics
        assert "credible_interval" in metrics

    def test_compare_filters_by_criteria(self):
        """Compare filters to specified criteria."""
        plugin = ScoreBayesianBaselinePlugin(criteria=["quality"])

        baseline = make_payload({"quality": [1.0, 2.0, 3.0], "accuracy": [1.0, 2.0, 3.0]})
        variant = make_payload({"quality": [2.0, 3.0, 4.0], "accuracy": [2.0, 3.0, 4.0]})

        result = plugin.compare(baseline, variant)

        assert "quality" in result
        assert "accuracy" not in result

    def test_compare_skips_insufficient_samples(self):
        """Compare skips criteria with insufficient samples."""
        plugin = ScoreBayesianBaselinePlugin(min_samples=10)

        baseline = make_payload({"quality": [1.0, 2.0, 3.0]})
        variant = make_payload({"quality": [2.0, 3.0, 4.0]})

        result = plugin.compare(baseline, variant)

        assert "quality" not in result

    def test_compare_high_probability_when_variant_better(self):
        """Probability is high when variant clearly better."""
        plugin = ScoreBayesianBaselinePlugin()

        # Use varied scores to avoid zero variance edge case
        baseline = make_payload({"quality": [1.0, 1.5, 2.0, 1.2, 1.8]})
        variant = make_payload({"quality": [10.0, 10.5, 11.0, 10.2, 10.8]})

        result = plugin.compare(baseline, variant)

        assert "quality" in result
        assert result["quality"]["prob_variant_gt_baseline"] > 0.99


class TestScoreBayesianBaselinePluginSchema:
    """Tests for schema declarations."""

    def test_has_name(self):
        """Plugin has name attribute."""
        assert ScoreBayesianBaselinePlugin.name == "score_bayes"

    def test_has_config_schema(self):
        """Plugin has config_schema."""
        assert hasattr(ScoreBayesianBaselinePlugin, "config_schema")
        schema = ScoreBayesianBaselinePlugin.config_schema
        assert schema["type"] == "object"
        assert "credible_interval" in schema["properties"]

    def test_has_input_schema(self):
        """Plugin has input_schema."""
        assert hasattr(ScoreBayesianBaselinePlugin, "input_schema")
        assert ScoreBayesianBaselinePlugin.input_schema["type"] == "object"

    def test_has_output_schema(self):
        """Plugin has output_schema."""
        assert hasattr(ScoreBayesianBaselinePlugin, "output_schema")
        assert ScoreBayesianBaselinePlugin.output_schema["type"] == "object"


class TestScoreBayesianBaselinePluginRegistration:
    """Tests for plugin registration."""

    def test_plugin_registered(self):
        """Plugin is registered with baseline factory."""
        from elspeth.core.sda.plugin_registry import create_baseline_plugin

        plugin = create_baseline_plugin({
            "name": "score_bayes",
            "options": {"credible_interval": 0.90},
        })

        assert isinstance(plugin, ScoreBayesianBaselinePlugin)
        assert plugin._ci == 0.90
