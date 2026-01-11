# tests/plugins/test_data_manipulation_integration.py
"""Integration tests for data manipulation plugins in realistic pipelines."""

from elspeth.core.sda.result_aggregator import ResultAggregator
from elspeth.plugins.transforms.aggregate_data_reshape import AggregateDataReshape
from elspeth.plugins.transforms.field_collector import FieldCollector
from elspeth.plugins.transforms.field_expander import FieldExpander
from elspeth.plugins.transforms.row_data_reshape import RowDataReshape


def test_row_reshape_then_collect_pipeline():
    """Test: Row reshape → Field collector → Analysis."""
    # Setup: Reshape rows, then collect

    row_reshape = RowDataReshape({
        "operations": [
            {"flatten": "metadata"},
            {"rename": {
                "metadata.confidence": "confidence",
                "metadata.sample_size": "n"
            }},
            {"exclude_fields": ["prompt", "response", "metadata"]}
        ]
    })

    collector = FieldCollector({"output_key": "collected"})

    # Use RowProcessor-style execution (not testing here, just simulating)
    # In real execution, RowProcessor would call row_reshape.transform()

    # Simulate row processing
    raw_rows = [
        {
            "id": 1,
            "score": 0.75,
            "prompt": "Q1",
            "response": "A1",
            "metadata": {"confidence": 0.95, "sample_size": 100}
        },
        {
            "id": 2,
            "score": 0.82,
            "prompt": "Q2",
            "response": "A2",
            "metadata": {"confidence": 0.88, "sample_size": 120}
        }
    ]

    # Apply row reshape
    reshaped_rows = []
    for row in raw_rows:
        reshaped = row_reshape.transform(row, context={})
        reshaped_rows.append(reshaped)

    # Collect reshaped rows
    collection = collector.aggregate(reshaped_rows, aggregates={})

    # Verify collection
    assert collection["id"] == [1, 2]
    assert collection["score"] == [0.75, 0.82]
    assert collection["confidence"] == [0.95, 0.88]
    assert collection["n"] == [100, 120]
    assert "prompt" not in collection
    assert "metadata" not in collection


def test_collect_then_aggregate_reshape_pipeline():
    """Test: Field collector → Aggregate reshape → Analysis."""
    collector = FieldCollector({"output_key": "raw_collection"})

    aggregate_reshape = AggregateDataReshape({
        "input_key": "raw_collection",
        "output_key": "clean_collection",
        "operations": [
            {"exclude_fields": ["prompt", "response"]},
            {"rename": {"baseline_score": "baseline", "variant_score": "variant"}},
            {"filter_fields": ["id", "baseline", "variant"]}
        ]
    })

    aggregator = ResultAggregator(aggregation_plugins=[collector, aggregate_reshape])

    # Add results
    aggregator.add_result({
        "id": 1,
        "baseline_score": 0.75,
        "variant_score": 0.82,
        "prompt": "Q1",
        "response": "A1"
    }, row_index=0)

    aggregator.add_result({
        "id": 2,
        "baseline_score": 0.68,
        "variant_score": 0.71,
        "prompt": "Q2",
        "response": "A2"
    }, row_index=1)

    payload = aggregator.build_payload(security_level=None, early_stop_reason=None)

    # Check clean collection
    clean = payload["aggregates"]["clean_collection"]
    assert list(clean.keys()) == ["id", "baseline", "variant"]
    assert clean["id"] == [1, 2]
    assert clean["baseline"] == [0.75, 0.68]


def test_collect_reshape_expand_round_trip():
    """Test: Collect → Reshape → Expand (should preserve data)."""
    collector = FieldCollector({"output_key": "collected"})

    aggregate_reshape = AggregateDataReshape({
        "input_key": "collected",
        "output_key": "reshaped",
        "operations": [
            {"filter_fields": ["id", "score", "delta"]}
        ]
    })

    expander = FieldExpander({"input_key": "reshaped"})

    aggregator = ResultAggregator(aggregation_plugins=[collector, aggregate_reshape, expander])

    # Original rows (with extra fields that will be filtered)
    original_rows = [
        {"id": 1, "score": 0.75, "delta": 0.15, "extra": "removed"},
        {"id": 2, "score": 0.82, "delta": 0.22, "extra": "removed"}
    ]

    for i, row in enumerate(original_rows):
        aggregator.add_result(row, row_index=i)

    payload = aggregator.build_payload(security_level=None, early_stop_reason=None)

    # Check expanded rows (should match filtered original)
    expanded = payload["aggregates"]["field_expander"]

    expected = [
        {"id": 1, "score": 0.75, "delta": 0.15},
        {"id": 2, "score": 0.82, "delta": 0.22}
    ]

    assert expanded == expected


def test_multiple_reshape_operations_in_sequence():
    """Test multiple aggregate reshape operations."""
    collector = FieldCollector({"output_key": "raw"})

    reshape1 = AggregateDataReshape({
        "input_key": "raw",
        "output_key": "step1",
        "operations": [
            {"exclude_fields": ["prompt", "response"]}
        ]
    })

    reshape2 = AggregateDataReshape({
        "input_key": "step1",
        "output_key": "step2",
        "operations": [
            {"rename": {"baseline_score": "b", "variant_score": "v"}}
        ]
    })

    reshape3 = AggregateDataReshape({
        "input_key": "step2",
        "output_key": "final",
        "operations": [
            {"filter_fields": ["id", "b", "v"]}
        ]
    })

    aggregator = ResultAggregator(
        aggregation_plugins=[collector, reshape1, reshape2, reshape3]
    )

    aggregator.add_result({
        "id": 1,
        "baseline_score": 0.75,
        "variant_score": 0.82,
        "delta": 0.07,
        "prompt": "Q1",
        "response": "A1"
    }, row_index=0)

    payload = aggregator.build_payload(security_level=None, early_stop_reason=None)

    # Should have all intermediate steps
    assert "raw" in payload["aggregates"]
    assert "step1" in payload["aggregates"]
    assert "step2" in payload["aggregates"]
    assert "final" in payload["aggregates"]

    # Final should be filtered and renamed
    final = payload["aggregates"]["final"]
    assert list(final.keys()) == ["id", "b", "v"]
    assert final["b"] == [0.75]
    assert final["v"] == [0.82]


def test_row_and_aggregate_reshape_together():
    """Test both row and aggregate reshape in same pipeline."""
    # This would be in a full SDA pipeline, but we simulate here

    row_reshape_config = {
        "operations": [
            {"flatten": "metadata"},
            {"exclude_fields": ["metadata"]}
        ]
    }

    # In real execution, this would be called by RowProcessor
    row_reshape = RowDataReshape(row_reshape_config)

    # Process rows
    rows = [
        {"id": 1, "score": 0.75, "metadata": {"confidence": 0.95}},
        {"id": 2, "score": 0.82, "metadata": {"confidence": 0.88}}
    ]

    processed_rows = [row_reshape.transform(r, context={}) for r in rows]

    # Now aggregate and reshape
    collector = FieldCollector({"output_key": "collected"})
    aggregate_reshape = AggregateDataReshape({
        "input_key": "collected",
        "output_key": "final",
        "operations": [
            {"rename": {"metadata.confidence": "conf"}},
            {"filter_fields": ["id", "score", "conf"]}
        ]
    })

    aggregator = ResultAggregator(aggregation_plugins=[collector, aggregate_reshape])

    for i, row in enumerate(processed_rows):
        aggregator.add_result(row, row_index=i)

    payload = aggregator.build_payload(security_level=None, early_stop_reason=None)

    final = payload["aggregates"]["final"]
    assert "conf" in final
    assert "metadata.confidence" not in final
    assert len(final["id"]) == 2


def test_documented_example_from_technical_spec():
    """Test example from technical spec (nested metadata flattening)."""
    # This example is from 2025-11-20-schema-system-technical-spec.md

    row_reshape = RowDataReshape({
        "operations": [
            {"flatten": "metadata"},
            {"rename": {
                "metadata.confidence": "confidence",
                "metadata.sample_size": "sample_size"
            }},
            {"exclude_fields": ["metadata"]}
        ]
    })

    # Input row (from CSV)
    input_row = {
        "id": 1,
        "baseline_score": 0.75,
        "variant_score": 0.82,
        "metadata": {"confidence": 0.95, "sample_size": 100}
    }

    # Apply reshape
    output_row = row_reshape.transform(input_row, context={})

    # Should match documented output
    expected = {
        "id": 1,
        "baseline_score": 0.75,
        "variant_score": 0.82,
        "confidence": 0.95,
        "sample_size": 100
    }

    assert output_row == expected
