"""Tests for ThresholdEarlyStopPlugin."""

from __future__ import annotations

import pytest

from elspeth.plugins.transforms.early_stop import ThresholdEarlyStopPlugin


class TestThresholdEarlyStopPluginInit:
    """Tests for ThresholdEarlyStopPlugin initialization."""

    def test_init_with_required_params(self):
        """Initializes with required parameters."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0.8)

        assert plugin._metric == "score"
        assert plugin._threshold == 0.8
        assert plugin._comparison == "gte"
        assert plugin._min_rows == 1

    def test_init_with_all_params(self):
        """Initializes with all parameters."""
        plugin = ThresholdEarlyStopPlugin(
            metric="metrics.accuracy",
            threshold=0.95,
            comparison="gt",
            min_rows=10,
            label="high_accuracy",
        )

        assert plugin._metric == "metrics.accuracy"
        assert plugin._threshold == 0.95
        assert plugin._comparison == "gt"
        assert plugin._min_rows == 10
        assert plugin._label == "high_accuracy"

    def test_init_requires_metric(self):
        """Requires metric parameter."""
        with pytest.raises(ValueError, match="requires a 'metric'"):
            ThresholdEarlyStopPlugin(metric="", threshold=0.8)

    def test_init_requires_valid_threshold(self):
        """Requires valid numeric threshold."""
        with pytest.raises(ValueError, match="Invalid threshold"):
            ThresholdEarlyStopPlugin(metric="score", threshold="not_a_number")

    def test_init_accepts_string_threshold(self):
        """Accepts numeric string as threshold."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold="0.8")

        assert plugin._threshold == 0.8

    def test_init_defaults_invalid_comparison_to_gte(self):
        """Invalid comparison defaults to 'gte'."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.8,
            comparison="invalid",
        )

        assert plugin._comparison == "gte"

    def test_init_normalizes_comparison_case(self):
        """Normalizes comparison to lowercase."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.8,
            comparison="GTE",
        )

        assert plugin._comparison == "gte"

    def test_init_min_rows_minimum_is_one(self):
        """Min rows has minimum value of 1."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.8,
            min_rows=0,
        )

        assert plugin._min_rows == 1

        plugin2 = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.8,
            min_rows=-5,
        )

        assert plugin2._min_rows == 1


class TestThresholdEarlyStopPluginCheck:
    """Tests for check method."""

    def test_returns_none_when_metric_not_found(self):
        """Returns None when metric path not found."""
        plugin = ThresholdEarlyStopPlugin(metric="missing", threshold=0.8)

        result = plugin.check({"metrics": {"score": 0.9}})

        assert result is None

    def test_returns_none_for_non_numeric_value(self):
        """Returns None for non-numeric metric value."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0.8)

        result = plugin.check({"metrics": {"score": "not_a_number"}})

        assert result is None

    def test_returns_none_before_min_rows(self):
        """Returns None before min_rows reached."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.8,
            min_rows=3,
        )

        # First two rows don't trigger even if threshold met
        result1 = plugin.check({"metrics": {"score": 0.9}})
        result2 = plugin.check({"metrics": {"score": 0.95}})

        assert result1 is None
        assert result2 is None
        assert plugin._rows_observed == 2

    def test_triggers_after_min_rows(self):
        """Triggers after min_rows reached."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.8,
            min_rows=2,
        )

        # First row doesn't trigger
        plugin.check({"metrics": {"score": 0.9}})
        # Second row triggers
        result = plugin.check({"metrics": {"score": 0.85}})

        assert result is not None
        assert result["value"] == 0.85
        assert result["threshold"] == 0.8
        assert result["rows_observed"] == 2


class TestThresholdComparisons:
    """Tests for different comparison operators."""

    def test_gte_comparison(self):
        """Tests >= comparison."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.8,
            comparison="gte",
        )

        # Below threshold
        assert plugin.check({"metrics": {"score": 0.79}}) is None
        # At threshold
        plugin.reset()
        result = plugin.check({"metrics": {"score": 0.8}})
        assert result is not None
        assert result["comparison"] == "gte"

    def test_gt_comparison(self):
        """Tests > comparison."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.8,
            comparison="gt",
        )

        # At threshold (should not trigger for >)
        assert plugin.check({"metrics": {"score": 0.8}}) is None
        # Above threshold
        plugin.reset()
        result = plugin.check({"metrics": {"score": 0.81}})
        assert result is not None
        assert result["comparison"] == "gt"

    def test_lte_comparison(self):
        """Tests <= comparison."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.5,
            comparison="lte",
        )

        # Above threshold
        assert plugin.check({"metrics": {"score": 0.51}}) is None
        # At threshold
        plugin.reset()
        result = plugin.check({"metrics": {"score": 0.5}})
        assert result is not None
        assert result["comparison"] == "lte"

    def test_lt_comparison(self):
        """Tests < comparison."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.5,
            comparison="lt",
        )

        # At threshold (should not trigger for <)
        assert plugin.check({"metrics": {"score": 0.5}}) is None
        # Below threshold
        plugin.reset()
        result = plugin.check({"metrics": {"score": 0.49}})
        assert result is not None
        assert result["comparison"] == "lt"


class TestNestedMetricExtraction:
    """Tests for nested metric path extraction."""

    def test_extracts_top_level_metric(self):
        """Extracts top-level metric."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0.8)

        result = plugin.check({"metrics": {"score": 0.9}})

        assert result is not None
        assert result["value"] == 0.9

    def test_extracts_nested_metric(self):
        """Extracts nested metric path."""
        plugin = ThresholdEarlyStopPlugin(metric="quality.accuracy", threshold=0.8)

        result = plugin.check({
            "metrics": {
                "quality": {"accuracy": 0.95}
            }
        })

        assert result is not None
        assert result["value"] == 0.95
        assert result["metric"] == "quality.accuracy"

    def test_extracts_deeply_nested_metric(self):
        """Extracts deeply nested metric path."""
        plugin = ThresholdEarlyStopPlugin(
            metric="experiment.variant_a.score.mean",
            threshold=0.7,
        )

        result = plugin.check({
            "metrics": {
                "experiment": {
                    "variant_a": {
                        "score": {"mean": 0.8}
                    }
                }
            }
        })

        assert result is not None
        assert result["value"] == 0.8

    def test_returns_none_for_incomplete_path(self):
        """Returns None when nested path is incomplete."""
        plugin = ThresholdEarlyStopPlugin(metric="a.b.c", threshold=0.8)

        result = plugin.check({"metrics": {"a": {"b": {}}}})

        assert result is None


class TestMetadataAttachment:
    """Tests for metadata in trigger results."""

    def test_includes_label_when_configured(self):
        """Includes label in result when configured."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.8,
            label="accuracy_threshold",
        )

        result = plugin.check({"metrics": {"score": 0.9}})

        assert result["label"] == "accuracy_threshold"

    def test_includes_provided_metadata(self):
        """Includes provided metadata in result."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0.8)

        result = plugin.check(
            {"metrics": {"score": 0.9}},
            metadata={"experiment": "test", "run_id": 123},
        )

        assert result["experiment"] == "test"
        assert result["run_id"] == 123

    def test_result_keys_not_overwritten_by_metadata(self):
        """Result keys are not overwritten by metadata."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0.8)

        result = plugin.check(
            {"metrics": {"score": 0.9}},
            metadata={"value": "should_not_replace", "threshold": "no"},
        )

        assert result["value"] == 0.9
        assert result["threshold"] == 0.8


class TestStateReset:
    """Tests for state reset between checks."""

    def test_reset_clears_row_count(self):
        """Reset clears row count."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0.8)

        plugin.check({"metrics": {"score": 0.5}})
        plugin.check({"metrics": {"score": 0.6}})
        assert plugin._rows_observed == 2

        plugin.reset()

        assert plugin._rows_observed == 0

    def test_reset_clears_triggered_reason(self):
        """Reset clears triggered reason."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0.8)

        # Trigger
        result1 = plugin.check({"metrics": {"score": 0.9}})
        assert result1 is not None
        assert plugin._triggered_reason is not None

        plugin.reset()

        assert plugin._triggered_reason is None

    def test_reset_allows_retrigger(self):
        """Reset allows re-triggering."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0.8)

        # First trigger
        result1 = plugin.check({"metrics": {"score": 0.9}})
        assert result1 is not None

        plugin.reset()

        # Can trigger again
        result2 = plugin.check({"metrics": {"score": 0.85}})
        assert result2 is not None
        assert result2["value"] == 0.85


class TestTriggeredStatePersistence:
    """Tests for triggered state persistence."""

    def test_returns_same_reason_after_triggered(self):
        """Returns same reason after initial trigger."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0.8)

        # Trigger
        result1 = plugin.check({"metrics": {"score": 0.9}})

        # Subsequent checks return same reason
        result2 = plugin.check({"metrics": {"score": 0.5}})  # Different value
        result3 = plugin.check({"metrics": {"score": 0.3}})

        assert result1 == result2 == result3
        assert result2["value"] == 0.9  # Original trigger value preserved


class TestMissingMetricsHandling:
    """Tests for missing metrics handling."""

    def test_handles_missing_metrics_key(self):
        """Handles record without metrics key."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0.8)

        result = plugin.check({})

        assert result is None

    def test_handles_none_metrics(self):
        """Handles None metrics value."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0.8)

        result = plugin.check({"metrics": None})

        assert result is None

    def test_increments_row_count_even_without_value(self):
        """Row count increments even when metric not found."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=0.8,
            min_rows=3,
        )

        # Two records without the metric
        plugin.check({"metrics": {"other": 1}})
        plugin.check({"metrics": {"other": 2}})

        # Metric not found so no increment
        assert plugin._rows_observed == 0


class TestPluginRegistration:
    """Tests for plugin registration."""

    def test_plugin_registered_with_factory(self):
        """Plugin is registered with factory."""
        from elspeth.core.sda.plugin_registry import create_halt_condition_plugin

        plugin = create_halt_condition_plugin({
            "name": "threshold",
            "options": {
                "metric": "score",
                "threshold": 0.9,
                "comparison": "gte",
            },
        })

        assert isinstance(plugin, ThresholdEarlyStopPlugin)
        assert plugin._metric == "score"
        assert plugin._threshold == 0.9

    def test_plugin_validates_options(self):
        """Plugin validates options via schema."""
        from elspeth.core.sda.plugin_registry import validate_early_stop_plugin_definition

        # Valid definition should not raise
        validate_early_stop_plugin_definition({
            "name": "threshold",
            "options": {
                "metric": "score",
                "threshold": 0.9,
            },
        })


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_threshold(self):
        """Works with zero threshold."""
        plugin = ThresholdEarlyStopPlugin(metric="score", threshold=0)

        result = plugin.check({"metrics": {"score": 0.1}})

        assert result is not None
        assert result["threshold"] == 0.0

    def test_negative_threshold(self):
        """Works with negative threshold."""
        plugin = ThresholdEarlyStopPlugin(
            metric="delta",
            threshold=-0.5,
            comparison="lt",
        )

        result = plugin.check({"metrics": {"delta": -0.6}})

        assert result is not None
        assert result["value"] == -0.6

    def test_integer_metric_value(self):
        """Works with integer metric values."""
        plugin = ThresholdEarlyStopPlugin(metric="count", threshold=10)

        result = plugin.check({"metrics": {"count": 15}})

        assert result is not None
        assert result["value"] == 15.0

    def test_very_large_threshold(self):
        """Works with very large threshold."""
        plugin = ThresholdEarlyStopPlugin(
            metric="score",
            threshold=1e10,
            comparison="gte",
        )

        result = plugin.check({"metrics": {"score": 1e11}})

        assert result is not None

    def test_very_small_threshold(self):
        """Works with very small threshold."""
        plugin = ThresholdEarlyStopPlugin(
            metric="error",
            threshold=1e-10,
            comparison="lte",
        )

        result = plugin.check({"metrics": {"error": 1e-11}})

        assert result is not None
