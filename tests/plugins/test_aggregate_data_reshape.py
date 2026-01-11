import pytest

from elspeth.plugins.transforms.aggregate_data_reshape import AggregateDataReshape


def test_aggregate_data_reshape_applies_operations_to_collection():
    """AggregateDataReshape should apply operations to collection data."""
    config = {
        "input_key": "collected",
        "output_key": "reshaped",
        "operations": [
            {"exclude_fields": ["prompt", "response"]}
        ]
    }

    plugin = AggregateDataReshape(config)

    # Simulated aggregates dict
    aggregates = {
        "collected": {
            "id": [1, 2],
            "score": [0.75, 0.82],
            "prompt": ["Q1", "Q2"],
            "response": ["A1", "A2"]
        }
    }

    result = plugin.aggregate(results=[], aggregates=aggregates)

    expected = {
        "id": [1, 2],
        "score": [0.75, 0.82]
    }

    assert result == expected


def test_aggregate_data_reshape_reads_from_input_key():
    """AggregateDataReshape should read collection from input_key."""
    config = {
        "input_key": "raw_data",
        "output_key": "processed",
        "operations": [
            {"filter_fields": ["id", "score"]}
        ]
    }

    plugin = AggregateDataReshape(config)

    aggregates = {
        "raw_data": {
            "id": [1, 2],
            "score": [0.75, 0.82],
            "extra": [10, 20]
        }
    }

    result = plugin.aggregate(results=[], aggregates=aggregates)

    assert "id" in result
    assert "score" in result
    assert "extra" not in result


def test_aggregate_data_reshape_config_schema():
    """AggregateDataReshape should have required config schema."""
    from elspeth.core.validation import validate_schema

    # Valid config
    valid_config = {
        "input_key": "collected",
        "output_key": "reshaped",
        "operations": [
            {"exclude_fields": ["prompt"]}
        ]
    }
    errors = list(validate_schema(valid_config, AggregateDataReshape.config_schema))
    assert len(errors) == 0

    # Invalid config (missing input_key)
    invalid_config = {
        "output_key": "reshaped",
        "operations": []
    }
    errors = list(validate_schema(invalid_config, AggregateDataReshape.config_schema))
    assert len(errors) > 0


def test_aggregate_data_reshape_has_schemas():
    """AggregateDataReshape should have input and output schemas."""
    assert hasattr(AggregateDataReshape, 'config_schema')
    assert hasattr(AggregateDataReshape, 'input_schema')
    assert hasattr(AggregateDataReshape, 'output_schema')

    # Should accept collection type
    assert AggregateDataReshape.input_schema['type'] == 'collection'
    assert AggregateDataReshape.output_schema['type'] == 'collection'


def test_aggregate_data_reshape_validates_input_key():
    """AggregateDataReshape should validate input_key exists."""
    config = {
        "input_key": "nonexistent",
        "output_key": "reshaped",
        "operations": [{"filter_fields": ["id"]}]
    }

    plugin = AggregateDataReshape(config)

    aggregates = {"other_key": {"id": [1, 2]}}

    with pytest.raises(KeyError) as exc_info:
        plugin.aggregate(results=[], aggregates=aggregates)

    error_msg = str(exc_info.value)
    assert "nonexistent" in error_msg
    assert "not found" in error_msg.lower()


def test_aggregate_data_reshape_validates_unsupported_operation():
    """AggregateDataReshape should reject operations that don't make sense for collections."""
    config = {
        "input_key": "collected",
        "output_key": "reshaped",
        "operations": [
            {"flatten": "metadata"}  # Doesn't make sense for collection
        ]
    }

    plugin = AggregateDataReshape(config)

    aggregates = {
        "collected": {"id": [1, 2], "score": [0.75, 0.82]}
    }

    with pytest.raises(ValueError) as exc_info:
        plugin.aggregate(results=[], aggregates=aggregates)

    error_msg = str(exc_info.value)
    assert "not supported" in error_msg.lower()
    assert "collection-level" in error_msg.lower()


def test_aggregate_data_reshape_pipeline_with_field_collector():
    """Test realistic pipeline: FieldCollector → AggregateDataReshape."""
    from elspeth.core.sda.result_aggregator import ResultAggregator
    from elspeth.plugins.transforms.field_collector import FieldCollector

    collector_config = {"output_key": "raw_data"}
    collector = FieldCollector(collector_config)

    reshape_config = {
        "input_key": "raw_data",
        "output_key": "clean_data",
        "operations": [
            {"exclude_fields": ["prompt", "response", "retry"]},
            {"rename": {"baseline_score": "baseline", "variant_score": "variant"}},
            {"filter_fields": ["id", "baseline", "variant", "delta"]}
        ]
    }
    reshape = AggregateDataReshape(reshape_config)

    aggregator = ResultAggregator(aggregation_plugins=[collector, reshape])

    # Add row results
    aggregator.add_result({
        "id": 1,
        "baseline_score": 0.75,
        "variant_score": 0.82,
        "delta": 0.07,
        "prompt": "Question 1",
        "response": "Answer 1",
        "retry": {"attempts": 0}
    }, row_index=0)

    aggregator.add_result({
        "id": 2,
        "baseline_score": 0.68,
        "variant_score": 0.71,
        "delta": 0.03,
        "prompt": "Question 2",
        "response": "Answer 2",
        "retry": {"attempts": 1}
    }, row_index=1)

    # Build payload
    payload = aggregator.build_payload(security_level=None, early_stop_reason=None)

    # Check raw_data collection exists
    assert "raw_data" in payload["aggregates"]
    raw = payload["aggregates"]["raw_data"]
    assert "prompt" in raw  # Still has prompt

    # Check clean_data collection
    assert "clean_data" in payload["aggregates"]
    clean = payload["aggregates"]["clean_data"]

    # Should only have renamed/filtered fields
    assert list(clean.keys()) == ["id", "baseline", "variant", "delta"]
    assert clean["id"] == [1, 2]
    assert clean["baseline"] == [0.75, 0.68]
    assert clean["variant"] == [0.82, 0.71]
    assert clean["delta"] == [0.07, 0.03]
