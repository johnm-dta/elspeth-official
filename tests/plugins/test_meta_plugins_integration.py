"""Integration tests for meta-plugins in full pipeline."""

from elspeth.plugins.transforms.field_collector import FieldCollector
from elspeth.plugins.transforms.field_expander import FieldExpander


class MockStatsAnalyzer:
    """Mock analyzer that reads from collection."""
    def __init__(self, config):
        self.config = config
        self.name = "stats_analyzer"
        self.input_schema = {
            "type": "collection",
            "item_schema": {"type": "object"}
        }
        self.output_schema = {
            "type": "object",
            "properties": {
                "mean": {"type": "number"},
                "count": {"type": "integer"}
            }
        }

    def aggregate(self, results, aggregates):
        """Analyze collected data."""
        input_key = self.config['input_key']
        collection = aggregates[input_key]

        # Calculate mean of scores
        scores = collection.get('score', [])
        if not scores:
            return {"mean": 0.0, "count": 0}

        mean = sum(scores) / len(scores)
        return {"mean": mean, "count": len(scores)}


def test_row_to_collection_to_analysis_pipeline():
    """Test full row→collection→analysis pipeline."""
    from elspeth.core.sda.result_aggregator import ResultAggregator

    # Create pipeline: FieldCollector → StatsAnalyzer
    collector = FieldCollector({"output_key": "collected_scores"})
    analyzer = MockStatsAnalyzer({"input_key": "collected_scores"})

    aggregator = ResultAggregator(aggregation_plugins=[collector, analyzer])

    # Add row results
    aggregator.add_result({"id": 1, "score": 0.75}, row_index=0)
    aggregator.add_result({"id": 2, "score": 0.82}, row_index=1)
    aggregator.add_result({"id": 3, "score": 0.91}, row_index=2)

    # Build payload
    payload = aggregator.build_payload(
        security_level=None,
        early_stop_reason=None
    )

    # Check collected data
    assert "collected_scores" in payload["aggregates"]
    collected = payload["aggregates"]["collected_scores"]
    assert collected["score"] == [0.75, 0.82, 0.91]
    assert collected["id"] == [1, 2, 3]

    # Check analyzed data
    assert "stats_analyzer" in payload["aggregates"]
    stats = payload["aggregates"]["stats_analyzer"]
    assert stats["count"] == 3
    assert abs(stats["mean"] - 0.8267) < 0.001  # Average of 0.75, 0.82, 0.91


def test_row_to_collection_to_row_pipeline():
    """Test row→collection→row pipeline with FieldExpander."""
    from elspeth.core.sda.result_aggregator import ResultAggregator

    # Create pipeline: FieldCollector → FieldExpander
    collector = FieldCollector({"output_key": "collected"})
    expander = FieldExpander({"input_key": "collected"})

    aggregator = ResultAggregator(aggregation_plugins=[collector, expander])

    # Add row results
    original_rows = [
        {"id": 1, "score": 0.75, "delta": 0.15},
        {"id": 2, "score": 0.82, "delta": 0.22},
        {"id": 3, "score": 0.91, "delta": 0.08}
    ]

    for i, row in enumerate(original_rows):
        aggregator.add_result(row, row_index=i)

    # Build payload
    payload = aggregator.build_payload(
        security_level=None,
        early_stop_reason=None
    )

    # Check collected data
    assert "collected" in payload["aggregates"]

    # Check expanded data (should be array of rows)
    assert "field_expander" in payload["aggregates"]
    expanded = payload["aggregates"]["field_expander"]
    assert isinstance(expanded, list)
    assert len(expanded) == 3
    assert expanded == original_rows


def test_multiple_collectors_with_different_keys():
    """Test multiple FieldCollectors with different output keys."""
    from elspeth.core.sda.result_aggregator import ResultAggregator

    # Two collectors with different exclusions
    collector1 = FieldCollector({
        "output_key": "scores_only",
        "exclude_fields": ["name", "metadata"]
    })

    collector2 = FieldCollector({
        "output_key": "names_only",
        "exclude_fields": ["id", "score"]
    })

    aggregator = ResultAggregator(aggregation_plugins=[collector1, collector2])

    # Add results
    aggregator.add_result({"id": 1, "score": 0.75, "name": "Alice", "metadata": {}}, row_index=0)
    aggregator.add_result({"id": 2, "score": 0.82, "name": "Bob", "metadata": {}}, row_index=1)

    payload = aggregator.build_payload(
        security_level=None,
        early_stop_reason=None
    )

    # Check scores_only collection
    scores_only = payload["aggregates"]["scores_only"]
    assert "id" in scores_only
    assert "score" in scores_only
    assert "name" not in scores_only
    assert "metadata" not in scores_only

    # Check names_only collection
    names_only = payload["aggregates"]["names_only"]
    assert "name" in names_only
    assert "metadata" in names_only
    assert "id" not in names_only
    assert "score" not in names_only
