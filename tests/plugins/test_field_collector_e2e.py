"""End-to-end test for FieldCollector with real SDA result structure."""


from elspeth.core.sda.result_aggregator import ResultAggregator
from elspeth.plugins.transforms.field_collector import FieldCollector
from elspeth.plugins.transforms.metrics import ScoreRecommendationAnalyzer, ScoreStatsAnalyzer


def test_field_collector_with_real_sda_results():
    """
    Integration test: FieldCollector flattens nested metrics from real SDA results.

    This test replicates the real SDA pipeline structure where:
    1. RowProcessor creates results with structure {row, response, metrics}
    2. FieldCollector flattens metrics to top-level arrays
    3. Analyzers consume the flattened metric fields
    """
    # Real SDA result structure (from RowProcessor)
    sda_results = [
        {
            "row": {"id": "APP001", "text": "Sample application 1"},
            "response": {
                "content": "Score: 3/4",
                "model": "gpt-4",
                "tokens": 150
            },
            "metrics": {
                "extracted_score": 0.75,
                "score_flags": True,
                "delta": 0.15,
                "confidence": 0.92
            }
        },
        {
            "row": {"id": "APP002", "text": "Sample application 2"},
            "response": {
                "content": "Score: 3.5/4",
                "model": "gpt-4",
                "tokens": 142
            },
            "metrics": {
                "extracted_score": 0.875,
                "score_flags": True,
                "delta": 0.22,
                "confidence": 0.95
            }
        },
        {
            "row": {"id": "APP003", "text": "Sample application 3"},
            "response": {
                "content": "Score: 4/4",
                "model": "gpt-4",
                "tokens": 138
            },
            "metrics": {
                "extracted_score": 1.0,
                "score_flags": False,
                "delta": 0.08,
                "confidence": 0.98
            }
        }
    ]

    # Setup pipeline: FieldCollector → ScoreStatsAnalyzer
    collector = FieldCollector({
        "output_key": "collected_metrics",
        "exclude_fields": ["row", "response"]  # Exclude nested objects
    })

    stats_analyzer = ScoreStatsAnalyzer(
        input_key="collected_metrics",
        source_field="extracted_score"
    )

    aggregator = ResultAggregator(
        aggregation_plugins=[collector, stats_analyzer]
    )

    # Add results
    for i, result in enumerate(sda_results):
        aggregator.add_result(result, row_index=i)

    # Build payload
    payload = aggregator.build_payload(
        security_level=None,
        early_stop_reason=None
    )

    # Verify FieldCollector flattened metrics correctly
    assert "collected_metrics" in payload["aggregates"]
    collection = payload["aggregates"]["collected_metrics"]

    # Should have flattened metric fields as arrays
    assert "extracted_score" in collection
    assert "score_flags" in collection
    assert "delta" in collection
    assert "confidence" in collection

    # Verify values
    assert collection["extracted_score"] == [0.75, 0.875, 1.0]
    assert collection["score_flags"] == [True, True, False]
    assert collection["delta"] == [0.15, 0.22, 0.08]
    assert collection["confidence"] == [0.92, 0.95, 0.98]

    # Should NOT have nested objects (excluded)
    assert "row" not in collection
    assert "response" not in collection

    # Should NOT have metrics as nested dict
    assert "metrics" not in collection

    # Verify ScoreStatsAnalyzer could read the flattened data
    # The analyzer is registered as "score_stats" (via register_aggregation_transform)
    assert "score_stats" in payload["aggregates"]
    stats = payload["aggregates"]["score_stats"]

    assert "mean" in stats
    assert "std" in stats
    assert "count" in stats
    assert stats["count"] == 3
    assert 0.87 < stats["mean"] < 0.88  # approx 0.875


def test_field_collector_with_recommendation_analyzer():
    """
    Integration test: FieldCollector → ScoreRecommendationAnalyzer.

    Tests the complete pipeline that was previously failing.
    """
    # Real SDA results with score metrics
    sda_results = [
        {
            "row": {"id": f"APP{i:03d}"},
            "response": {"content": f"Response {i}"},
            "metrics": {
                "extracted_score": 0.75 + (i * 0.05),  # Scores: 0.75, 0.80, 0.85, 0.90, 0.95
                "score_flags": True
            }
        }
        for i in range(5)
    ]

    # Setup pipeline
    collector = FieldCollector({
        "output_key": "collected",
        "exclude_fields": ["row", "response"]
    })

    recommendation_analyzer = ScoreRecommendationAnalyzer(
        input_key="collected",
        source_field="extracted_score",
        flag_field="score_flags"
    )

    aggregator = ResultAggregator(
        aggregation_plugins=[collector, recommendation_analyzer]
    )

    # Add results
    for i, result in enumerate(sda_results):
        aggregator.add_result(result, row_index=i)

    # Build payload
    payload = aggregator.build_payload(
        security_level=None,
        early_stop_reason=None
    )

    # Verify collection
    assert "collected" in payload["aggregates"]
    collection = payload["aggregates"]["collected"]
    assert "extracted_score" in collection
    assert "score_flags" in collection

    # Verify recommendation analyzer ran successfully
    # The analyzer is registered as "score_recommendation" (via register_aggregation_transform)
    assert "score_recommendation" in payload["aggregates"]
    recommendation = payload["aggregates"]["score_recommendation"]

    # Should have recommendation fields
    assert "recommendation" in recommendation
    assert "sample_size" in recommendation
    assert "effect_size" in recommendation
    assert recommendation["sample_size"] == 5
    assert recommendation["recommendation"] in ["stop", "continue"]


def test_field_collector_mixed_top_level_and_metrics():
    """
    Test FieldCollector with fields at both top-level and in metrics.

    Top-level fields should take precedence.
    """
    results = [
        {
            "id": 1,
            "top_level_field": "value1",
            "response": {"content": "Response 1"},
            "metrics": {
                "extracted_score": 0.75,
                "metric_only_field": "metric1"
            }
        },
        {
            "id": 2,
            "top_level_field": "value2",
            "response": {"content": "Response 2"},
            "metrics": {
                "extracted_score": 0.82,
                "metric_only_field": "metric2"
            }
        }
    ]

    collector = FieldCollector({"output_key": "collected"})
    collection = collector.aggregate(results, aggregates={})

    # Should have both top-level and metric fields
    assert "id" in collection
    assert "top_level_field" in collection
    assert "response" in collection
    assert "extracted_score" in collection
    assert "metric_only_field" in collection

    # Verify values
    assert collection["id"] == [1, 2]
    assert collection["top_level_field"] == ["value1", "value2"]
    assert collection["extracted_score"] == [0.75, 0.82]
    assert collection["metric_only_field"] == ["metric1", "metric2"]


def test_field_collector_handles_missing_metrics():
    """
    Test that FieldCollector uses None for missing metric fields.
    """
    collector = FieldCollector({"output_key": "collected"})

    # Row 0 has field in metrics, Row 1 missing it
    results = [
        {
            "id": 1,
            "metrics": {"extracted_score": 0.75}
        },
        {
            "id": 2,
            "metrics": {"other_field": 0.82}  # Missing extracted_score!
        }
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should collect all fields with None for missing values
    assert collection["id"] == [1, 2]
    assert collection["extracted_score"] == [0.75, None]
    assert collection["other_field"] == [None, 0.82]
