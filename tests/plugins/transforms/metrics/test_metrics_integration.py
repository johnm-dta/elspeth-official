"""Integration tests for refactored metrics system."""

import pytest

from elspeth.plugins.transforms.field_collector import FieldCollector
from elspeth.plugins.transforms.metrics import (
    ScoreAgreementAnalyzer,
    ScoreAssumptionsBaselinePlugin,
    ScoreBayesianBaselinePlugin,
    ScoreCliffsDeltaPlugin,
    ScoreDeltaBaselinePlugin,
    ScoreDistributionAnalyzer,
    ScoreExtractorPlugin,
    ScorePowerAnalyzer,
    ScorePracticalBaselinePlugin,
    ScoreRecommendationAnalyzer,
    ScoreSignificanceBaselinePlugin,
    ScoreStatsAnalyzer,
    ScoreVariantRankingAnalyzer,
)


def test_complete_metrics_pipeline():
    """Test complete pipeline: extractor → collector → analyzers."""

    # Step 1: Extract scores (row-mode)
    extractor = ScoreExtractorPlugin(key="score")

    rows_with_responses = [
        {"id": 1, "default": {"metrics": {"score": 0.75}}},
        {"id": 2, "default": {"metrics": {"score": 0.82}}},
        {"id": 3, "default": {"metrics": {"score": 0.91}}},
    ]

    extracted_rows = []
    for row in rows_with_responses:
        # Extract responses from row
        row_responses = {k: v for k, v in row.items() if k != "id"}
        result = extractor.transform(row, row_responses)
        # Merge result with original row
        merged = {**row, **result}
        extracted_rows.append(merged)

    # Each row should have scores field with default key
    assert all("scores" in row for row in extracted_rows)
    assert all("default" in row["scores"] for row in extracted_rows)

    # Step 2: Collect scores (row → collection)
    # FieldCollector now recursively flattens nested metric dicts
    # So metrics.scores.default → "default" array
    collector = FieldCollector({"output_key": "collected"})
    collection = collector.aggregate(extracted_rows, aggregates={})

    # Collection should have flattened the nested scores dict
    # metrics.scores.default → "default" field in collection
    assert "default" in collection
    assert len(collection["default"]) == 3
    assert collection["default"] == [0.75, 0.82, 0.91]

    # Step 3: Analyze statistics (collection-mode)
    stats_analyzer = ScoreStatsAnalyzer(
        input_key="collected",
        source_field="default"
    )

    aggregates = {"collected": collection}
    stats = stats_analyzer.aggregate([], aggregates)

    assert stats["count"] == 3
    assert stats["mean"] > 0.8
    assert stats["std"] > 0

    # Step 4: Generate recommendation (collection-mode)
    rec_analyzer = ScoreRecommendationAnalyzer(
        input_key="collected",
        source_field="default",
        min_samples=2
    )

    recommendation = rec_analyzer.aggregate([], aggregates)

    assert "recommendation" in recommendation
    assert recommendation["recommendation"] in ["continue", "stop", "inconclusive"]


def test_multiple_analyzers_same_collection():
    """Multiple analyzers can read from same FieldCollector collection."""

    # Setup collection
    collector = FieldCollector({"output_key": "data"})
    rows = [{"score": s} for s in [0.6, 0.7, 0.8, 0.9]]
    collection = collector.aggregate(rows, aggregates={})
    aggregates = {"data": collection}

    # Multiple analyzers read same collection
    stats = ScoreStatsAnalyzer(input_key="data", source_field="score")
    recommendation = ScoreRecommendationAnalyzer(
        input_key="data",
        source_field="score",
        min_samples=2
    )

    stats_result = stats.aggregate([], aggregates)
    rec_result = recommendation.aggregate([], aggregates)

    # Both analyzers work on same data
    assert stats_result["count"] == 4
    assert "recommendation" in rec_result


def test_backward_compatibility_aliases():
    """Old class names should still work (deprecated)."""
    # Should be same class
    from elspeth.plugins.transforms.metrics import (
        ScoreAgreementAggregator,
        ScoreAgreementAnalyzer,
        ScoreDistributionAggregator,
        ScoreDistributionAnalyzer,
        ScorePowerAggregator,
        ScorePowerAnalyzer,
        ScoreRecommendationAggregator,
        ScoreRecommendationAnalyzer,
        ScoreStatsAggregator,  # Alias for ScoreStatsAnalyzer
        ScoreStatsAnalyzer,
        ScoreVariantRankingAggregator,
        ScoreVariantRankingAnalyzer,
    )

    assert ScoreStatsAggregator is ScoreStatsAnalyzer
    assert ScoreRecommendationAggregator is ScoreRecommendationAnalyzer
    assert ScoreVariantRankingAggregator is ScoreVariantRankingAnalyzer
    assert ScoreAgreementAggregator is ScoreAgreementAnalyzer
    assert ScorePowerAggregator is ScorePowerAnalyzer
    assert ScoreDistributionAggregator is ScoreDistributionAnalyzer


def test_all_13_plugins_importable():
    """All 13 metrics plugins should be importable from metrics package."""
    from elspeth.plugins.transforms.metrics import (
        ScoreAgreementAnalyzer,
        ScoreAssumptionsBaselinePlugin,
        ScoreBayesianBaselinePlugin,
        ScoreCliffsDeltaPlugin,
        # 6 comparisons
        ScoreDeltaBaselinePlugin,
        ScoreDistributionAnalyzer,
        # 1 extractor
        ScoreExtractorPlugin,
        ScorePowerAnalyzer,
        ScorePracticalBaselinePlugin,
        ScoreRecommendationAnalyzer,
        ScoreSignificanceBaselinePlugin,
        # 6 analyzers
        ScoreStatsAnalyzer,
        ScoreVariantRankingAnalyzer,
    )

    # All should be classes
    assert all(isinstance(cls, type) for cls in [
        ScoreExtractorPlugin,
        ScoreDeltaBaselinePlugin,
        ScoreCliffsDeltaPlugin,
        ScoreAssumptionsBaselinePlugin,
        ScorePracticalBaselinePlugin,
        ScoreSignificanceBaselinePlugin,
        ScoreBayesianBaselinePlugin,
        ScoreStatsAnalyzer,
        ScoreRecommendationAnalyzer,
        ScoreVariantRankingAnalyzer,
        ScoreAgreementAnalyzer,
        ScorePowerAnalyzer,
        ScoreDistributionAnalyzer,
    ])


@pytest.mark.parametrize("analyzer_class", [
    ScoreStatsAnalyzer,
    ScoreRecommendationAnalyzer,
    ScoreVariantRankingAnalyzer,
])
def test_analyzer_requires_field_collector(analyzer_class):
    """Analyzers should give helpful error if collection missing."""
    analyzer = analyzer_class(input_key="missing", source_field="score")

    with pytest.raises(KeyError) as exc_info:
        analyzer.aggregate([], aggregates={})

    error_msg = str(exc_info.value)
    assert "FieldCollector" in error_msg  # Suggests solution


def test_all_plugins_have_schemas():
    """All 13 plugins should have required schema attributes."""
    from elspeth.plugins.transforms.metrics import (
        ScoreAgreementAnalyzer,
        ScoreAssumptionsBaselinePlugin,
        ScoreBayesianBaselinePlugin,
        ScoreCliffsDeltaPlugin,
        ScoreDeltaBaselinePlugin,
        ScoreDistributionAnalyzer,
        ScoreExtractorPlugin,
        ScorePowerAnalyzer,
        ScorePracticalBaselinePlugin,
        ScoreRecommendationAnalyzer,
        ScoreSignificanceBaselinePlugin,
        ScoreStatsAnalyzer,
        ScoreVariantRankingAnalyzer,
    )

    all_plugins = [
        ScoreExtractorPlugin,
        ScoreDeltaBaselinePlugin,
        ScoreCliffsDeltaPlugin,
        ScoreAssumptionsBaselinePlugin,
        ScorePracticalBaselinePlugin,
        ScoreSignificanceBaselinePlugin,
        ScoreBayesianBaselinePlugin,
        ScoreStatsAnalyzer,
        ScoreRecommendationAnalyzer,
        ScoreVariantRankingAnalyzer,
        ScoreAgreementAnalyzer,
        ScorePowerAnalyzer,
        ScoreDistributionAnalyzer,
    ]

    for plugin_class in all_plugins:
        assert hasattr(plugin_class, 'config_schema'), f"{plugin_class.__name__} missing config_schema"
        assert hasattr(plugin_class, 'input_schema'), f"{plugin_class.__name__} missing input_schema"
        assert hasattr(plugin_class, 'output_schema'), f"{plugin_class.__name__} missing output_schema"

        # Verify schemas are dicts
        assert isinstance(plugin_class.config_schema, dict)
        assert isinstance(plugin_class.input_schema, dict)
        assert isinstance(plugin_class.output_schema, dict)

        # Verify input/output schemas have type
        assert 'type' in plugin_class.input_schema
        assert 'type' in plugin_class.output_schema


def test_row_plugins_have_object_schemas():
    """Row-mode plugins should have object-type schemas."""
    row_plugins = [
        ScoreExtractorPlugin,
        ScoreDeltaBaselinePlugin,
        ScoreCliffsDeltaPlugin,
        ScoreAssumptionsBaselinePlugin,
        ScorePracticalBaselinePlugin,
        ScoreSignificanceBaselinePlugin,
        ScoreBayesianBaselinePlugin,
    ]

    for plugin_class in row_plugins:
        assert plugin_class.input_schema['type'] == 'object'
        assert plugin_class.output_schema['type'] == 'object'


def test_analyzers_have_collection_schemas():
    """Analyzer plugins should have collection-type input schemas."""
    analyzers = [
        ScoreStatsAnalyzer,
        ScoreRecommendationAnalyzer,
        ScoreVariantRankingAnalyzer,
        ScoreAgreementAnalyzer,
        ScorePowerAnalyzer,
        ScoreDistributionAnalyzer,
    ]

    for analyzer_class in analyzers:
        assert analyzer_class.input_schema['type'] == 'collection'
        assert analyzer_class.output_schema['type'] == 'object'


def test_analyzers_require_input_key_in_config():
    """All aggregation analyzers should require input_key in their config schema."""
    from elspeth.core.validation import validate_schema

    # Note: ScoreDistributionAnalyzer has optional input_key (can be used as comparison plugin)
    analyzers = [
        ScoreStatsAnalyzer,
        ScoreRecommendationAnalyzer,
        ScoreVariantRankingAnalyzer,
        ScoreAgreementAnalyzer,
        ScorePowerAnalyzer,
    ]

    for analyzer_class in analyzers:
        # Config without input_key should fail validation
        invalid_config = {"source_field": "score"}
        errors = list(validate_schema(invalid_config, analyzer_class.config_schema))

        # Should have validation error
        assert len(errors) > 0, f"{analyzer_class.__name__} should require input_key"

        # Error should mention input_key
        error_messages = " ".join(str(e) for e in errors)
        assert "input_key" in error_messages.lower()


def test_extractor_to_multiple_analyzers_pipeline():
    """Test realistic pipeline with extractor feeding multiple analyzers."""

    # For simplicity, work with pre-extracted data
    # In real usage, extractor would process LLM responses
    rows = [
        {"id": 1, "quality_score": 0.8, "relevance_score": 0.7},
        {"id": 2, "quality_score": 0.9, "relevance_score": 0.85},
        {"id": 3, "quality_score": 0.75, "relevance_score": 0.8},
        {"id": 4, "quality_score": 0.95, "relevance_score": 0.9},
    ]

    # Step 1: Collect both score fields
    collector = FieldCollector({"output_key": "all_scores"})
    collection = collector.aggregate(rows, aggregates={})

    # Should have both score fields
    assert "quality_score" in collection
    assert "relevance_score" in collection

    # Step 2: Multiple analyzers on quality scores
    aggregates = {"all_scores": collection}

    stats = ScoreStatsAnalyzer(
        input_key="all_scores",
        source_field="quality_score"
    )

    recommendation = ScoreRecommendationAnalyzer(
        input_key="all_scores",
        source_field="quality_score",
        min_samples=2
    )

    stats_result = stats.aggregate([], aggregates)
    rec_result = recommendation.aggregate([], aggregates)

    # Verify stats
    assert stats_result["count"] == 4
    assert 0.8 < stats_result["mean"] < 0.9

    # Verify recommendation
    assert rec_result["recommendation"] in ["continue", "stop", "inconclusive"]
    assert rec_result["sample_size"] == 4


def test_error_messages_suggest_field_collector():
    """Error messages should suggest using FieldCollector when collection missing."""

    analyzer = ScoreStatsAnalyzer(
        input_key="nonexistent_collection",
        source_field="score"
    )

    with pytest.raises(KeyError) as exc_info:
        analyzer.aggregate([], aggregates={})

    error_msg = str(exc_info.value)
    assert "nonexistent_collection" in error_msg
    assert "not found in aggregates" in error_msg
    assert "FieldCollector" in error_msg
    assert "output_key" in error_msg


def test_analyzer_missing_field_helpful_error():
    """Analyzers should give helpful error when field not in collection."""

    # Collection without the field analyzer expects
    collector = FieldCollector({"output_key": "data"})
    rows = [{"score": 0.8}, {"score": 0.9}]
    collection = collector.aggregate(rows, aggregates={})

    analyzer = ScoreStatsAnalyzer(
        input_key="data",
        source_field="nonexistent_field"
    )

    with pytest.raises(ValueError) as exc_info:
        analyzer.aggregate([], aggregates={"data": collection})

    error_msg = str(exc_info.value)
    assert "nonexistent_field" in error_msg
    assert "not found in collection" in error_msg
    assert "Available fields" in error_msg
