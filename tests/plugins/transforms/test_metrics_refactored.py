"""Tests for refactored metrics analyzers using FieldCollector."""

import pytest

from elspeth.plugins.transforms.field_collector import FieldCollector
from elspeth.plugins.transforms.metrics import (
    ScoreAgreementAnalyzer,
    ScoreDistributionAnalyzer,
    ScorePowerAnalyzer,
    ScoreRecommendationAnalyzer,
    ScoreStatsAnalyzer,
    ScoreVariantRankingAnalyzer,
)


def test_score_stats_analyzer_reads_from_collection():
    """ScoreStatsAnalyzer should read collection from input_key."""
    # Setup: FieldCollector creates collection
    collector_config = {"output_key": "collected"}
    collector = FieldCollector(collector_config)

    rows = [
        {"id": 1, "extracted_score": 0.75},
        {"id": 2, "extracted_score": 0.82},
        {"id": 3, "extracted_score": 0.91},
    ]

    collection = collector.aggregate(rows, aggregates={})

    # ScoreStatsAnalyzer reads from collection
    analyzer_config = {
        "input_key": "collected",
        "source_field": "extracted_score"
    }

    analyzer = ScoreStatsAnalyzer(**analyzer_config)

    # Pass collection via aggregates dict
    aggregates = {"collected": collection}
    result = analyzer.aggregate([], aggregates)

    # Should compute stats from collected scores
    assert "mean" in result
    assert "std" in result
    assert "count" in result
    assert result["count"] == 3
    assert 0.82 < result["mean"] < 0.83  # approx 0.826


def test_score_stats_analyzer_has_collection_schema():
    """ScoreStatsAnalyzer should accept collection input."""
    assert hasattr(ScoreStatsAnalyzer, 'config_schema')
    assert hasattr(ScoreStatsAnalyzer, 'input_schema')
    assert hasattr(ScoreStatsAnalyzer, 'output_schema')

    # Input: collection type (from FieldCollector)
    assert ScoreStatsAnalyzer.input_schema['type'] == 'collection'

    # Output: object with stats
    assert ScoreStatsAnalyzer.output_schema['type'] == 'object'


def test_score_stats_analyzer_config_requires_input_key():
    """ScoreStatsAnalyzer config_schema should require input_key."""
    from elspeth.core.validation import validate_schema

    # Missing input_key
    invalid_config = {"source_field": "score"}
    errors = list(validate_schema(invalid_config, ScoreStatsAnalyzer.config_schema))
    assert len(errors) > 0
    assert "input_key" in str(errors[0])

    # Valid config
    valid_config = {"input_key": "collected", "source_field": "score"}
    errors = list(validate_schema(valid_config, ScoreStatsAnalyzer.config_schema))
    assert len(errors) == 0


def test_score_stats_analyzer_missing_input_key_error():
    """Should give helpful error if input_key not in aggregates."""
    analyzer = ScoreStatsAnalyzer(
        input_key="missing_key",
        source_field="score"
    )

    with pytest.raises(KeyError) as exc_info:
        analyzer.aggregate([], aggregates={"other_key": {}})

    error_msg = str(exc_info.value)
    assert "missing_key" in error_msg
    assert "not found in aggregates" in error_msg
    assert "Available keys" in error_msg
    assert "FieldCollector" in error_msg  # Suggests solution


def test_score_stats_analyzer_missing_field_error():
    """Should give helpful error if source_field not in collection."""
    analyzer = ScoreStatsAnalyzer(
        input_key="collected",
        source_field="missing_field"
    )

    collection = {"id": [1, 2, 3], "score": [0.7, 0.8, 0.9]}

    with pytest.raises(ValueError) as exc_info:
        analyzer.aggregate([], aggregates={"collected": collection})

    error_msg = str(exc_info.value)
    assert "missing_field" in error_msg
    assert "not found in collection" in error_msg
    assert "Available fields" in error_msg


# Task 3: Tests for 5 remaining analyzers


@pytest.mark.parametrize("analyzer_class", [
    ScoreRecommendationAnalyzer,
    ScoreVariantRankingAnalyzer,
    ScoreAgreementAnalyzer,
    ScorePowerAnalyzer,
    ScoreDistributionAnalyzer,
])
def test_analyzer_has_collection_input_schema(analyzer_class):
    """All analyzers should accept collection input."""
    assert hasattr(analyzer_class, 'input_schema')
    assert analyzer_class.input_schema['type'] == 'collection'


@pytest.mark.parametrize("analyzer_class", [
    ScoreRecommendationAnalyzer,
    ScoreVariantRankingAnalyzer,
    ScoreAgreementAnalyzer,
    ScorePowerAnalyzer,
    # Note: ScoreDistributionAnalyzer excluded - it's a comparison plugin with optional input_key
])
def test_analyzer_config_requires_input_key(analyzer_class):
    """All aggregation analyzer config_schemas should require input_key."""
    from elspeth.core.validation import validate_schema

    # Config without input_key should fail
    invalid_config = {}
    errors = list(validate_schema(invalid_config, analyzer_class.config_schema))

    # Should have validation error mentioning input_key
    assert any("input_key" in str(e) for e in errors)


def test_score_recommendation_analyzer_integration():
    """Test ScoreRecommendationAnalyzer with FieldCollector."""
    # Setup collection
    collector = FieldCollector({"output_key": "scores"})
    rows = [{"score": s} for s in [0.75, 0.80, 0.85]]
    collection = collector.aggregate(rows, aggregates={})

    # Analyze
    analyzer = ScoreRecommendationAnalyzer(
        input_key="scores",
        source_field="score",
        min_samples=2
    )

    result = analyzer.aggregate([], {"scores": collection})

    assert "recommendation" in result
    assert result["recommendation"] in ["continue", "stop", "inconclusive"]


def test_score_stats_analyzer_single_sample_std():
    """ScoreStatsAnalyzer should return 0.0 for std with single sample (not NaN)."""
    from elspeth.plugins.transforms.field_collector import FieldCollector
    from elspeth.plugins.transforms.metrics import ScoreStatsAnalyzer

    # Single sample results
    results = [
        {"score": 0.75}
    ]

    # Collect
    collector = FieldCollector({"output_key": "scores"})
    collection = collector.aggregate(results, aggregates={})

    # Analyze with ddof=1 (would produce NaN if not guarded)
    analyzer = ScoreStatsAnalyzer(
        input_key="scores",
        source_field="score",
        ddof=1
    )

    stats = analyzer.aggregate([], {"scores": collection})

    # Should return valid stats, not NaN
    assert stats["count"] == 1
    assert stats["mean"] == 0.75
    assert stats["std"] == 0.0  # Not NaN!
    assert stats["min"] == 0.75
    assert stats["max"] == 0.75
    assert stats["median"] == 0.75


def test_score_stats_analyzer_two_samples_std():
    """ScoreStatsAnalyzer should compute std normally with 2+ samples."""
    from elspeth.plugins.transforms.field_collector import FieldCollector
    from elspeth.plugins.transforms.metrics import ScoreStatsAnalyzer

    # Two sample results
    results = [
        {"score": 0.70},
        {"score": 0.80}
    ]

    # Collect
    collector = FieldCollector({"output_key": "scores"})
    collection = collector.aggregate(results, aggregates={})

    # Analyze with ddof=1
    analyzer = ScoreStatsAnalyzer(
        input_key="scores",
        source_field="score",
        ddof=1
    )

    stats = analyzer.aggregate([], {"scores": collection})

    # Should compute actual std (not 0.0)
    assert stats["count"] == 2
    assert stats["mean"] == 0.75
    assert stats["std"] > 0  # Should be ~0.0707


def test_score_delta_baseline_plugin_flat_schema():
    """ScoreDeltaBaselinePlugin should work with new flat schema (no criteria nesting)."""
    from elspeth.plugins.transforms.metrics.baseline_comparison import ScoreDeltaBaselinePlugin

    plugin = ScoreDeltaBaselinePlugin(metric="mean")

    # New flat schema (no "criteria" nesting)
    baseline = {
        "aggregates": {
            "score_stats": {
                "mean": 0.70,
                "std": 0.05,
                "count": 10
            }
        }
    }

    variant = {
        "aggregates": {
            "score_stats": {
                "mean": 0.85,
                "std": 0.06,
                "count": 12
            }
        }
    }

    result = plugin.compare(baseline, variant)

    # Should compute delta and ratio
    assert "delta" in result
    assert "ratio" in result
    assert "baseline_value" in result
    assert "variant_value" in result

    assert abs(result["delta"] - 0.15) < 0.0001  # 0.85 - 0.70
    assert abs(result["ratio"] - 1.214) < 0.01  # 0.85 / 0.70
    assert result["baseline_value"] == 0.70
    assert result["variant_value"] == 0.85


def test_score_stats_analyzer_filters_none_values():
    """ScoreStatsAnalyzer should filter out None values from collections."""
    from elspeth.plugins.transforms.field_collector import FieldCollector
    from elspeth.plugins.transforms.metrics import ScoreStatsAnalyzer

    # FieldCollector produces None for missing values
    results = [
        {"score": 0.75},
        {"score": None},  # Missing score
        {"score": 0.82},
        {"score": None},  # Missing score
        {"score": 0.90},
    ]

    # Collect
    collector = FieldCollector({"output_key": "scores"})
    collection = collector.aggregate(results, aggregates={})

    # Collection will have [0.75, None, 0.82, None, 0.90]
    assert collection["score"] == [0.75, None, 0.82, None, 0.90]

    # Analyze - should filter out Nones
    analyzer = ScoreStatsAnalyzer(
        input_key="scores",
        source_field="score",
        ddof=1
    )

    stats = analyzer.aggregate([], {"scores": collection})

    # Should compute stats on only the 3 valid scores
    assert stats["count"] == 3
    assert stats["missing_count"] == 2
    assert abs(stats["mean"] - 0.823) < 0.01  # (0.75 + 0.82 + 0.90) / 3
    assert stats["min"] == 0.75
    assert stats["max"] == 0.90


def test_score_stats_analyzer_all_none_values():
    """ScoreStatsAnalyzer handles all-None collections gracefully."""
    from elspeth.plugins.transforms.field_collector import FieldCollector
    from elspeth.plugins.transforms.metrics import ScoreStatsAnalyzer

    # All missing scores
    results = [
        {"score": None},
        {"score": None},
        {"score": None},
    ]

    collector = FieldCollector({"output_key": "scores"})
    collection = collector.aggregate(results, aggregates={})

    analyzer = ScoreStatsAnalyzer(
        input_key="scores",
        source_field="score",
        ddof=1
    )

    stats = analyzer.aggregate([], {"scores": collection})

    # Should return None stats with missing count
    assert stats["count"] == 0
    assert stats["missing_count"] == 3
    assert stats["mean"] is None
    assert stats["std"] is None
    assert stats["min"] is None
    assert stats["max"] is None


def test_score_stats_analyzer_mixed_none_and_nan():
    """ScoreStatsAnalyzer filters both None and NaN values."""
    import math

    from elspeth.plugins.transforms.field_collector import FieldCollector
    from elspeth.plugins.transforms.metrics import ScoreStatsAnalyzer

    # Mix of valid, None, and NaN
    results = [
        {"score": 0.75},
        {"score": None},
        {"score": math.nan},
        {"score": 0.85},
    ]

    collector = FieldCollector({"output_key": "scores"})
    collection = collector.aggregate(results, aggregates={})

    analyzer = ScoreStatsAnalyzer(
        input_key="scores",
        source_field="score",
        ddof=1
    )

    stats = analyzer.aggregate([], {"scores": collection})

    # Should only count the 2 valid scores
    assert stats["count"] == 2
    assert stats["missing_count"] == 2
    assert stats["mean"] == 0.80  # (0.75 + 0.85) / 2


def test_score_recommendation_analyzer_filters_none_values():
    """ScoreRecommendationAnalyzer should filter out None values from collections."""
    from elspeth.plugins.transforms.field_collector import FieldCollector
    from elspeth.plugins.transforms.metrics import ScoreRecommendationAnalyzer

    # Mix of valid and None values
    results = [
        {"score": 0.75},
        {"score": None},  # Missing score
        {"score": 0.82},
        {"score": None},  # Missing score
        {"score": 0.90},
    ]

    collector = FieldCollector({"output_key": "scores"})
    collection = collector.aggregate(results, aggregates={})

    analyzer = ScoreRecommendationAnalyzer(
        input_key="scores",
        source_field="score",
        min_samples=2
    )

    result = analyzer.aggregate([], {"scores": collection})

    # Should compute recommendation using only 3 valid scores
    assert "recommendation" in result
    assert result["recommendation"] in ["continue", "stop", "inconclusive"]
    assert result["sample_size"] == 3
    # Mean is ~0.823, which is above 0.5, so recommendation depends on improvement_margin


def test_score_recommendation_analyzer_all_none_values():
    """ScoreRecommendationAnalyzer handles all-None collections gracefully."""
    from elspeth.plugins.transforms.field_collector import FieldCollector
    from elspeth.plugins.transforms.metrics import ScoreRecommendationAnalyzer

    # All missing scores
    results = [
        {"score": None},
        {"score": None},
        {"score": None},
    ]

    collector = FieldCollector({"output_key": "scores"})
    collection = collector.aggregate(results, aggregates={})

    analyzer = ScoreRecommendationAnalyzer(
        input_key="scores",
        source_field="score",
        min_samples=2
    )

    result = analyzer.aggregate([], {"scores": collection})

    # Should return "continue" with sample_size 0 (insufficient samples)
    assert result["recommendation"] == "continue"
    assert result["sample_size"] == 0
    assert "reason" in result


def test_score_variant_ranking_analyzer_filters_none_values():
    """ScoreVariantRankingAnalyzer should filter out None values from collections."""
    from elspeth.plugins.transforms.field_collector import FieldCollector
    from elspeth.plugins.transforms.metrics import ScoreVariantRankingAnalyzer

    # Mix of valid and None values
    results = [
        {"score": 0.75},
        {"score": None},  # Missing score
        {"score": 0.82},
        {"score": None},  # Missing score
        {"score": 0.90},
    ]

    collector = FieldCollector({"output_key": "scores"})
    collection = collector.aggregate(results, aggregates={})

    analyzer = ScoreVariantRankingAnalyzer(
        input_key="scores",
        source_field="score",
        threshold=0.7
    )

    result = analyzer.aggregate([], {"scores": collection})

    # Should compute statistics using only 3 valid scores
    assert "samples" in result
    assert "mean" in result
    assert "median" in result
    assert result["samples"] == 3
    assert result["median"] == 0.82  # Median of [0.75, 0.82, 0.90]
    assert 0.82 < result["mean"] < 0.83  # approx 0.823


def test_score_variant_ranking_analyzer_all_none_values():
    """ScoreVariantRankingAnalyzer handles all-None collections gracefully."""
    from elspeth.plugins.transforms.field_collector import FieldCollector
    from elspeth.plugins.transforms.metrics import ScoreVariantRankingAnalyzer

    # All missing scores
    results = [
        {"score": None},
        {"score": None},
        {"score": None},
    ]

    collector = FieldCollector({"output_key": "scores"})
    collection = collector.aggregate(results, aggregates={})

    analyzer = ScoreVariantRankingAnalyzer(
        input_key="scores",
        source_field="score",
        threshold=0.7
    )

    result = analyzer.aggregate([], {"scores": collection})

    # Should return empty dict when all values are None
    assert result == {}


def test_score_agreement_analyzer_filters_none_values():
    """ScoreAgreementAnalyzer should filter out None values from collections."""
    from elspeth.plugins.transforms.field_collector import FieldCollector
    from elspeth.plugins.transforms.metrics import ScoreAgreementAnalyzer

    # Mix of valid and None values in multiple criteria
    results = [
        {"quality": 0.75, "accuracy": 0.80},
        {"quality": None, "accuracy": 0.85},  # Missing quality
        {"quality": 0.82, "accuracy": None},  # Missing accuracy
        {"quality": 0.90, "accuracy": 0.95},
    ]

    collector = FieldCollector({"output_key": "scores"})
    collection = collector.aggregate(results, aggregates={})

    analyzer = ScoreAgreementAnalyzer(
        input_key="scores",
        criteria=["quality", "accuracy"],
        min_items=2
    )

    result = analyzer.aggregate([], {"scores": collection})

    # Should compute agreement using valid values only
    # quality has 3 valid values, accuracy has 3 valid values (both >= min_items=2)
    assert "criteria" in result
    assert "cronbach_alpha" in result
    assert "average_correlation" in result
    assert set(result["criteria"]) == {"accuracy", "quality"}


def test_score_agreement_analyzer_insufficient_valid_values():
    """ScoreAgreementAnalyzer handles sparse data with insufficient valid pairs."""
    from elspeth.plugins.transforms.field_collector import FieldCollector
    from elspeth.plugins.transforms.metrics import ScoreAgreementAnalyzer

    # Most values are None
    results = [
        {"quality": 0.75, "accuracy": None},
        {"quality": None, "accuracy": None},
        {"quality": None, "accuracy": 0.85},
    ]

    collector = FieldCollector({"output_key": "scores"})
    collection = collector.aggregate(results, aggregates={})

    analyzer = ScoreAgreementAnalyzer(
        input_key="scores",
        criteria=["quality", "accuracy"],
        min_items=2
    )

    result = analyzer.aggregate([], {"scores": collection})

    # Should return empty dict when insufficient valid pairs
    # quality has 1 valid, accuracy has 1 valid (both < min_items=2)
    assert result == {}
