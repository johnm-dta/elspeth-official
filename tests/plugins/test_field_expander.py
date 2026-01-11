import pytest

from elspeth.plugins.transforms.field_expander import FieldExpander


def test_field_expander_transposes_columns_to_rows():
    """FieldExpander should transpose columnar data to row objects."""
    config = {"input_key": "collected"}

    expander = FieldExpander(config)

    # Simulated aggregates dict (from FieldCollector)
    aggregates = {
        "collected": {
            "id": [1, 2, 3],
            "score": [0.75, 0.82, 0.91],
            "delta": [0.15, 0.22, 0.08]
        }
    }

    # Execute aggregation
    rows = expander.aggregate(results=[], aggregates=aggregates)

    # Output: array of row objects
    expected = [
        {"id": 1, "score": 0.75, "delta": 0.15},
        {"id": 2, "score": 0.82, "delta": 0.22},
        {"id": 3, "score": 0.91, "delta": 0.08}
    ]

    assert rows == expected


def test_field_expander_config_schema():
    """FieldExpander should have required config schema."""
    from elspeth.core.validation import validate_schema

    # Valid config
    valid_config = {"input_key": "collected"}
    errors = list(validate_schema(valid_config, FieldExpander.config_schema))
    assert len(errors) == 0

    # Invalid config (missing input_key)
    invalid_config = {}
    errors = list(validate_schema(invalid_config, FieldExpander.config_schema))
    assert len(errors) > 0
    assert "required" in errors[0].message.lower()


def test_field_expander_has_schemas():
    """FieldExpander should have input and output schemas."""
    assert hasattr(FieldExpander, 'config_schema')
    assert hasattr(FieldExpander, 'input_schema')
    assert hasattr(FieldExpander, 'output_schema')

    # Input should be collection type
    assert FieldExpander.input_schema['type'] == 'collection'

    # Output should be array type
    assert FieldExpander.output_schema['type'] == 'array'


def test_field_expander_validates_input_key_exists():
    """FieldExpander should fail if input_key not in aggregates."""
    config = {"input_key": "missing_key"}
    expander = FieldExpander(config)

    aggregates = {
        "collected": {"score": [1, 2, 3]}
    }

    with pytest.raises(KeyError) as exc_info:
        expander.aggregate(results=[], aggregates=aggregates)

    error_msg = str(exc_info.value)
    assert "missing_key" in error_msg
    assert "not found" in error_msg.lower()


def test_field_expander_validates_array_lengths():
    """FieldExpander should detect inconsistent array lengths."""
    config = {"input_key": "collected"}
    expander = FieldExpander(config)

    # Inconsistent lengths: scores has 3 items, names has 2
    aggregates = {
        "collected": {
            "score": [0.75, 0.82, 0.91],
            "name": ["Alice", "Bob"]  # Different length!
        }
    }

    with pytest.raises(ValueError) as exc_info:
        expander.aggregate(results=[], aggregates=aggregates)

    error_msg = str(exc_info.value)
    assert "inconsistent" in error_msg.lower()
    assert "length" in error_msg.lower()


def test_field_collector_and_expander_round_trip():
    """Collect then expand should produce original data."""
    from elspeth.plugins.transforms.field_collector import FieldCollector

    # Original row data
    original_rows = [
        {"id": 1, "score": 0.75, "delta": 0.15},
        {"id": 2, "score": 0.82, "delta": 0.22},
        {"id": 3, "score": 0.91, "delta": 0.08}
    ]

    # Collect
    collector = FieldCollector({"output_key": "collected"})
    collection = collector.aggregate(original_rows, aggregates={})

    # Expand
    expander = FieldExpander({"input_key": "collected"})
    expanded_rows = expander.aggregate(results=[], aggregates={"collected": collection})

    # Should match original
    assert expanded_rows == original_rows


def test_field_expander_handles_empty_collection():
    """FieldExpander should handle empty collection gracefully."""
    config = {"input_key": "collected"}
    expander = FieldExpander(config)

    aggregates = {"collected": {}}

    rows = expander.aggregate(results=[], aggregates=aggregates)

    assert rows == []
