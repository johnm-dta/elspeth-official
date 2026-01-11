"""Integration tests for complex schema-driven pipelines."""

import pytest

from elspeth.plugins.transforms.field_collector import FieldCollector
from elspeth.plugins.transforms.field_expander import FieldExpander
from elspeth.plugins.transforms.metrics import (
    ScoreRecommendationAnalyzer,
    ScoreStatsAnalyzer,
)


class TestRowCollectionRowTransitions:
    """Test row→collection→row data flow transitions."""

    def test_basic_collect_expand_roundtrip(self):
        """Data survives collect→expand round-trip."""
        rows = [
            {"id": 1, "score": 0.8, "name": "alice"},
            {"id": 2, "score": 0.9, "name": "bob"},
            {"id": 3, "score": 0.7, "name": "carol"},
        ]

        # Row → Collection
        collector = FieldCollector({"output_key": "collected"})
        aggregates = {}
        collection = collector.aggregate(rows, aggregates)

        # Verify collection structure
        assert "score" in collection
        assert collection["score"] == [0.8, 0.9, 0.7]
        assert collection["name"] == ["alice", "bob", "carol"]

        # Collection → Row objects (via expander)
        expander = FieldExpander({"input_key": "collected"})
        aggregates["collected"] = collection
        expanded = expander.aggregate([], aggregates)

        # Expander returns list of row objects
        assert isinstance(expanded, list)
        assert len(expanded) == 3

        # Verify data round-trips correctly
        assert expanded[0] == {"id": 1, "score": 0.8, "name": "alice"}
        assert expanded[1] == {"id": 2, "score": 0.9, "name": "bob"}
        assert expanded[2] == {"id": 3, "score": 0.7, "name": "carol"}

    def test_multiple_collectors_different_keys(self):
        """Multiple FieldCollectors can write to different output_keys."""
        rows_a = [{"score": 0.8}, {"score": 0.9}]
        rows_b = [{"rating": 4.5}, {"rating": 4.8}]

        collector_a = FieldCollector({"output_key": "scores"})
        collector_b = FieldCollector({"output_key": "ratings"})

        aggregates = {}

        # Both collectors write to same aggregates dict
        collection_a = collector_a.aggregate(rows_a, aggregates)
        aggregates["scores"] = collection_a

        collection_b = collector_b.aggregate(rows_b, aggregates)
        aggregates["ratings"] = collection_b

        # Both collections available
        assert "scores" in aggregates
        assert "ratings" in aggregates
        assert aggregates["scores"]["score"] == [0.8, 0.9]
        assert aggregates["ratings"]["rating"] == [4.5, 4.8]


class TestMultiStageAnalyzerPipelines:
    """Test pipelines with multiple analyzer stages."""

    def test_collector_feeds_multiple_analyzers(self):
        """Single FieldCollector output feeds multiple analyzers."""
        rows = [
            {"score": 0.75},
            {"score": 0.82},
            {"score": 0.91},
            {"score": 0.88},
            {"score": 0.79},
        ]

        # Collect
        collector = FieldCollector({"output_key": "data"})
        aggregates = {}
        collection = collector.aggregate(rows, aggregates)
        aggregates["data"] = collection

        # Multiple analyzers read same collection
        stats = ScoreStatsAnalyzer(input_key="data", source_field="score")
        recommendation = ScoreRecommendationAnalyzer(
            input_key="data",
            source_field="score",
            min_samples=3
        )

        stats_result = stats.aggregate([], aggregates)
        rec_result = recommendation.aggregate([], aggregates)

        # Both get valid results
        assert stats_result["count"] == 5
        assert 0.8 < stats_result["mean"] < 0.85
        assert rec_result["recommendation"] in ["continue", "stop", "inconclusive"]
        assert rec_result["sample_size"] == 5

    def test_chained_analyzer_pipeline(self):
        """Analyzers can be chained in sequence."""
        rows = [{"delta": d} for d in [0.1, 0.15, 0.08, 0.12, 0.09]]

        collector = FieldCollector({"output_key": "deltas"})
        aggregates = {}
        collection = collector.aggregate(rows, aggregates)
        aggregates["deltas"] = collection

        # First analyzer: stats
        stats = ScoreStatsAnalyzer(input_key="deltas", source_field="delta")
        stats_result = stats.aggregate([], aggregates)

        # Stats can inform next stage
        assert stats_result["mean"] > 0
        assert stats_result["std"] > 0

        # Recommendation uses same collection
        rec = ScoreRecommendationAnalyzer(
            input_key="deltas",
            source_field="delta",
            min_samples=3
        )
        rec_result = rec.aggregate([], aggregates)

        assert "recommendation" in rec_result


class TestPipelineErrorPropagation:
    """Test error handling in multi-stage pipelines."""

    def test_missing_collection_key_error(self):
        """Analyzer gives clear error when input_key not in aggregates."""
        aggregates = {"other_key": {"score": [1, 2, 3]}}

        analyzer = ScoreStatsAnalyzer(
            input_key="missing_key",
            source_field="score"
        )

        with pytest.raises(KeyError) as exc_info:
            analyzer.aggregate([], aggregates)

        error_msg = str(exc_info.value)
        assert "missing_key" in error_msg
        assert "not found in aggregates" in error_msg
        assert "FieldCollector" in error_msg

    def test_missing_field_in_collection_error(self):
        """Analyzer gives clear error when field not in collection."""
        collection = {"score": [0.8, 0.9], "name": ["a", "b"]}
        aggregates = {"data": collection}

        analyzer = ScoreStatsAnalyzer(
            input_key="data",
            source_field="nonexistent"
        )

        with pytest.raises(ValueError) as exc_info:
            analyzer.aggregate([], aggregates)

        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg
        assert "not found in collection" in error_msg
        assert "Available fields" in error_msg

    def test_empty_collection_handling(self):
        """Pipeline handles empty data gracefully."""
        collector = FieldCollector({"output_key": "empty"})
        aggregates = {}
        collection = collector.aggregate([], aggregates)
        aggregates["empty"] = collection

        # Empty collection should be empty dict
        assert collection == {}

        # Analyzer should handle gracefully
        analyzer = ScoreStatsAnalyzer(
            input_key="empty",
            source_field="score"
        )

        with pytest.raises(ValueError) as exc_info:
            analyzer.aggregate([], aggregates)

        # Should mention empty or missing field
        assert "score" in str(exc_info.value).lower()
