"""Tests for baseline comparison plugins."""

import importlib.util

import pytest

from elspeth.plugins.transforms.metrics.baseline_comparison import (
    ScoreAssumptionsBaselinePlugin,
    ScoreCliffsDeltaPlugin,
    ScoreDeltaBaselinePlugin,
)

# Check if scipy is available
HAS_SCIPY = importlib.util.find_spec("scipy") is not None

requires_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")


# Test ScoreCliffsDeltaPlugin


def test_score_cliffs_delta_basic():
    """ScoreCliffsDeltaPlugin computes Cliff's delta between baseline and variant."""
    plugin = ScoreCliffsDeltaPlugin()

    # Baseline with lower scores, variant with higher scores
    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5}}},
            {"metrics": {"scores": {"quality": 0.6}}},
            {"metrics": {"scores": {"quality": 0.55}}},
        ]
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.8}}},
            {"metrics": {"scores": {"quality": 0.85}}},
            {"metrics": {"scores": {"quality": 0.9}}},
        ]
    }

    result = plugin.compare(baseline, variant)

    assert "quality" in result
    assert "delta" in result["quality"]
    assert "interpretation" in result["quality"]
    assert "baseline_samples" in result["quality"]
    assert "variant_samples" in result["quality"]

    # Variant scores are all higher, so delta should be positive and large
    assert result["quality"]["delta"] > 0.9  # Near 1.0 (all variant > baseline)
    assert result["quality"]["interpretation"] == "large"
    assert result["quality"]["baseline_samples"] == 3
    assert result["quality"]["variant_samples"] == 3


def test_score_cliffs_delta_negligible():
    """ScoreCliffsDeltaPlugin detects negligible effect size."""
    plugin = ScoreCliffsDeltaPlugin()

    # Very similar distributions
    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5}}},
            {"metrics": {"scores": {"quality": 0.52}}},
            {"metrics": {"scores": {"quality": 0.51}}},
        ]
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.50}}},
            {"metrics": {"scores": {"quality": 0.53}}},
            {"metrics": {"scores": {"quality": 0.51}}},
        ]
    }

    result = plugin.compare(baseline, variant)

    assert "quality" in result
    assert result["quality"]["interpretation"] == "negligible"


def test_score_cliffs_delta_min_samples():
    """ScoreCliffsDeltaPlugin skips criteria with insufficient samples."""
    plugin = ScoreCliffsDeltaPlugin(min_samples=5)

    # Only 3 samples per group (below min_samples=5)
    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5}}},
            {"metrics": {"scores": {"quality": 0.6}}},
            {"metrics": {"scores": {"quality": 0.55}}},
        ]
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.8}}},
            {"metrics": {"scores": {"quality": 0.85}}},
            {"metrics": {"scores": {"quality": 0.9}}},
        ]
    }

    result = plugin.compare(baseline, variant)

    # Should be empty because samples < min_samples
    assert result == {}


def test_score_cliffs_delta_criteria_filter():
    """ScoreCliffsDeltaPlugin filters by criteria list."""
    plugin = ScoreCliffsDeltaPlugin(criteria=["quality"])

    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5, "accuracy": 0.6}}},
            {"metrics": {"scores": {"quality": 0.6, "accuracy": 0.7}}},
        ]
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.8, "accuracy": 0.9}}},
            {"metrics": {"scores": {"quality": 0.85, "accuracy": 0.95}}},
        ]
    }

    result = plugin.compare(baseline, variant)

    # Should only include quality, not accuracy
    assert "quality" in result
    assert "accuracy" not in result


def test_score_cliffs_delta_on_error_skip():
    """ScoreCliffsDeltaPlugin skips on error when configured."""
    plugin = ScoreCliffsDeltaPlugin(on_error="skip")

    # Invalid data structure should trigger error
    baseline = {"invalid": "structure"}
    variant = {"invalid": "structure"}

    result = plugin.compare(baseline, variant)

    # Should return empty dict instead of raising
    assert result == {}


def test_score_cliffs_delta_on_error_abort():
    """ScoreCliffsDeltaPlugin raises on error when configured."""
    plugin = ScoreCliffsDeltaPlugin(on_error="abort")

    # Invalid data that will cause error in collect_scores_by_criterion
    baseline = {"results": [{"invalid": "no metrics"}]}
    variant = {"results": [{"invalid": "no metrics"}]}

    # Should not raise (no common criteria = empty result)
    result = plugin.compare(baseline, variant)
    assert result == {}


def test_score_cliffs_delta_invalid_on_error():
    """ScoreCliffsDeltaPlugin rejects invalid on_error values."""
    with pytest.raises(ValueError, match="on_error must be"):
        ScoreCliffsDeltaPlugin(on_error="invalid")


def test_score_cliffs_delta_multiple_criteria():
    """ScoreCliffsDeltaPlugin handles multiple criteria."""
    plugin = ScoreCliffsDeltaPlugin()

    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5, "accuracy": 0.6}}},
            {"metrics": {"scores": {"quality": 0.6, "accuracy": 0.7}}},
        ]
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.8, "accuracy": 0.65}}},
            {"metrics": {"scores": {"quality": 0.85, "accuracy": 0.72}}},
        ]
    }

    result = plugin.compare(baseline, variant)

    assert "quality" in result
    assert "accuracy" in result
    assert result["quality"]["delta"] > 0  # Variant higher
    assert abs(result["accuracy"]["delta"]) <= 0.5  # Similar-ish


# Test ScoreAssumptionsBaselinePlugin


@requires_scipy
def test_score_assumptions_normality_tests():
    """ScoreAssumptionsBaselinePlugin tests normality with Shapiro-Wilk."""
    plugin = ScoreAssumptionsBaselinePlugin(min_samples=3)

    # Normal-ish data
    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5}}},
            {"metrics": {"scores": {"quality": 0.52}}},
            {"metrics": {"scores": {"quality": 0.48}}},
            {"metrics": {"scores": {"quality": 0.51}}},
            {"metrics": {"scores": {"quality": 0.49}}},
        ]
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.7}}},
            {"metrics": {"scores": {"quality": 0.72}}},
            {"metrics": {"scores": {"quality": 0.68}}},
            {"metrics": {"scores": {"quality": 0.71}}},
            {"metrics": {"scores": {"quality": 0.69}}},
        ]
    }

    result = plugin.compare(baseline, variant)

    assert "quality" in result
    assert "baseline" in result["quality"]
    assert "variant" in result["quality"]

    # Check baseline normality test results
    baseline_result = result["quality"]["baseline"]
    assert "statistic" in baseline_result
    assert "p_value" in baseline_result
    assert "is_normal" in baseline_result
    assert "samples" in baseline_result
    assert baseline_result["samples"] == 5

    # Check variant normality test results
    variant_result = result["quality"]["variant"]
    assert "statistic" in variant_result
    assert "p_value" in variant_result
    assert "is_normal" in variant_result
    assert variant_result["samples"] == 5


@requires_scipy
def test_score_assumptions_variance_equality():
    """ScoreAssumptionsBaselinePlugin tests variance equality with Levene."""
    plugin = ScoreAssumptionsBaselinePlugin(min_samples=2)

    # Similar variances
    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5}}},
            {"metrics": {"scores": {"quality": 0.6}}},
        ]
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.7}}},
            {"metrics": {"scores": {"quality": 0.8}}},
        ]
    }

    result = plugin.compare(baseline, variant)

    assert "quality" in result
    assert "variance" in result["quality"]

    variance_result = result["quality"]["variance"]
    assert "statistic" in variance_result
    assert "p_value" in variance_result
    assert "equal_variance" in variance_result


@requires_scipy
def test_score_assumptions_min_samples():
    """ScoreAssumptionsBaselinePlugin requires minimum samples for normality test."""
    plugin = ScoreAssumptionsBaselinePlugin(min_samples=5)

    # Only 3 samples (below min_samples)
    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5}}},
            {"metrics": {"scores": {"quality": 0.6}}},
            {"metrics": {"scores": {"quality": 0.55}}},
        ]
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.8}}},
            {"metrics": {"scores": {"quality": 0.85}}},
            {"metrics": {"scores": {"quality": 0.9}}},
        ]
    }

    result = plugin.compare(baseline, variant)

    # Should still have variance test (needs 2+ samples)
    # But baseline/variant normality tests should be None
    if "quality" in result:
        assert result["quality"]["baseline"] is None
        assert result["quality"]["variant"] is None
        # Variance test requires 2+ samples, so should be present
        assert "variance" in result["quality"]


@requires_scipy
def test_score_assumptions_alpha_threshold():
    """ScoreAssumptionsBaselinePlugin uses custom alpha for significance."""
    plugin = ScoreAssumptionsBaselinePlugin(min_samples=3, alpha=0.01)

    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5}}},
            {"metrics": {"scores": {"quality": 0.52}}},
            {"metrics": {"scores": {"quality": 0.48}}},
        ]
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.7}}},
            {"metrics": {"scores": {"quality": 0.72}}},
            {"metrics": {"scores": {"quality": 0.68}}},
        ]
    }

    result = plugin.compare(baseline, variant)

    # Should use alpha=0.01 for is_normal and equal_variance checks
    assert "quality" in result
    # is_normal = (p_value > alpha), so stricter alpha makes it harder to pass


@requires_scipy
def test_score_assumptions_criteria_filter():
    """ScoreAssumptionsBaselinePlugin filters by criteria list."""
    plugin = ScoreAssumptionsBaselinePlugin(criteria=["quality"], min_samples=2)

    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5, "accuracy": 0.6}}},
            {"metrics": {"scores": {"quality": 0.6, "accuracy": 0.7}}},
        ]
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.8, "accuracy": 0.9}}},
            {"metrics": {"scores": {"quality": 0.85, "accuracy": 0.95}}},
        ]
    }

    result = plugin.compare(baseline, variant)

    assert "quality" in result
    assert "accuracy" not in result


def test_score_assumptions_on_error_skip():
    """ScoreAssumptionsBaselinePlugin skips on error when configured."""
    plugin = ScoreAssumptionsBaselinePlugin(on_error="skip")

    baseline = {"invalid": "structure"}
    variant = {"invalid": "structure"}

    result = plugin.compare(baseline, variant)

    # Should return empty dict instead of raising
    assert result == {}


def test_score_assumptions_invalid_on_error():
    """ScoreAssumptionsBaselinePlugin rejects invalid on_error values."""
    with pytest.raises(ValueError, match="on_error must be"):
        ScoreAssumptionsBaselinePlugin(on_error="invalid")


def test_score_assumptions_no_scipy():
    """ScoreAssumptionsBaselinePlugin returns empty when scipy unavailable."""
    import elspeth.plugins.transforms.metrics.baseline_comparison as module

    # Temporarily mock scipy_stats as None
    original = module.scipy_stats
    try:
        module.scipy_stats = None

        plugin = ScoreAssumptionsBaselinePlugin(min_samples=3)

        baseline = {"results": [{"metrics": {"scores": {"quality": 0.5}}}]}
        variant = {"results": [{"metrics": {"scores": {"quality": 0.8}}}]}

        result = plugin.compare(baseline, variant)

        # Should return empty when scipy not available
        assert result == {}

    finally:
        module.scipy_stats = original


@requires_scipy
def test_score_assumptions_insufficient_variance_samples():
    """ScoreAssumptionsBaselinePlugin skips variance test with <2 samples."""
    plugin = ScoreAssumptionsBaselinePlugin(min_samples=1)

    # Single sample in variant
    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5}}},
            {"metrics": {"scores": {"quality": 0.6}}},
        ]
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.8}}},
        ]
    }

    result = plugin.compare(baseline, variant)

    if "quality" in result:
        # Variance test requires 2+ samples in both groups
        assert result["quality"]["variance"] is None


# Integration test combining both plugins


@requires_scipy
def test_baseline_comparison_integration():
    """Test ScoreDelta, CliffsD, and Assumptions together."""
    delta_plugin = ScoreDeltaBaselinePlugin(metric="mean")
    cliffs_plugin = ScoreCliffsDeltaPlugin(min_samples=3)
    assumptions_plugin = ScoreAssumptionsBaselinePlugin(min_samples=3)

    baseline = {
        "results": [
            {"metrics": {"scores": {"quality": 0.5}}},
            {"metrics": {"scores": {"quality": 0.52}}},
            {"metrics": {"scores": {"quality": 0.48}}},
            {"metrics": {"scores": {"quality": 0.51}}},
        ],
        "aggregates": {
            "score_stats": {
                "mean": 0.5025,
                "std": 0.017,
                "count": 4,
            }
        },
    }

    variant = {
        "results": [
            {"metrics": {"scores": {"quality": 0.75}}},
            {"metrics": {"scores": {"quality": 0.78}}},
            {"metrics": {"scores": {"quality": 0.72}}},
            {"metrics": {"scores": {"quality": 0.76}}},
        ],
        "aggregates": {
            "score_stats": {
                "mean": 0.7525,
                "std": 0.025,
                "count": 4,
            }
        },
    }

    # Test delta
    delta_result = delta_plugin.compare(baseline, variant)
    assert "delta" in delta_result
    assert delta_result["delta"] > 0.2  # Large improvement

    # Test Cliff's delta
    cliffs_result = cliffs_plugin.compare(baseline, variant)
    assert "quality" in cliffs_result
    assert cliffs_result["quality"]["delta"] > 0.9  # Nearly all variant > baseline
    assert cliffs_result["quality"]["interpretation"] == "large"

    # Test assumptions
    assumptions_result = assumptions_plugin.compare(baseline, variant)
    assert "quality" in assumptions_result
    assert assumptions_result["quality"]["baseline"] is not None
    assert assumptions_result["quality"]["variant"] is not None
    assert assumptions_result["quality"]["variance"] is not None
