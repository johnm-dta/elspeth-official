# tests/plugins/test_row_data_reshape.py
import pytest

from elspeth.plugins.transforms.row_data_reshape import RowDataReshape


def test_row_data_reshape_applies_single_operation():
    """RowDataReshape should apply single operation to row."""
    config = {
        "operations": [
            {"exclude_fields": ["prompt", "response"]}
        ]
    }

    plugin = RowDataReshape(config)

    row = {
        "id": 1,
        "score": 0.75,
        "prompt": "Question 1",
        "response": "Answer 1"
    }

    result = plugin.transform(row, context={})

    expected = {"id": 1, "score": 0.75}

    assert result == expected


def test_row_data_reshape_applies_multiple_operations():
    """RowDataReshape should apply operations in sequence."""
    config = {
        "operations": [
            {"flatten": "metadata"},
            {"rename": {
                "metadata.confidence": "confidence",
                "metadata.sample_size": "sample_size"
            }},
            {"exclude_fields": ["metadata"]}
        ]
    }

    plugin = RowDataReshape(config)

    row = {
        "id": 1,
        "score": 0.75,
        "metadata": {
            "confidence": 0.95,
            "sample_size": 100
        }
    }

    result = plugin.transform(row, context={})

    expected = {
        "id": 1,
        "score": 0.75,
        "confidence": 0.95,
        "sample_size": 100
    }

    assert result == expected


def test_row_data_reshape_config_schema():
    """RowDataReshape should have required config schema."""
    from elspeth.core.validation import validate_schema

    # Valid config
    valid_config = {
        "operations": [
            {"exclude_fields": ["prompt"]}
        ]
    }
    errors = list(validate_schema(valid_config, RowDataReshape.config_schema))
    assert len(errors) == 0

    # Invalid config (missing operations)
    invalid_config = {}
    errors = list(validate_schema(invalid_config, RowDataReshape.config_schema))
    assert len(errors) > 0
    assert "required" in errors[0].message.lower()


def test_row_data_reshape_has_schemas():
    """RowDataReshape should have input and output schemas."""
    assert hasattr(RowDataReshape, 'config_schema')
    assert hasattr(RowDataReshape, 'input_schema')
    assert hasattr(RowDataReshape, 'output_schema')

    # Should accept objects (row-mode)
    assert RowDataReshape.input_schema['type'] == 'object'
    assert RowDataReshape.output_schema['type'] == 'object'


def test_row_data_reshape_validates_operation_format():
    """RowDataReshape should validate operation format."""
    config = {
        "operations": [
            "invalid"  # Should be dict, not string
        ]
    }

    plugin = RowDataReshape(config)
    row = {"id": 1}

    with pytest.raises(ValueError) as exc_info:
        plugin.transform(row, context={})

    error_msg = str(exc_info.value)
    assert "not a dict" in error_msg.lower()


def test_row_data_reshape_validates_single_key():
    """RowDataReshape should require single key per operation."""
    config = {
        "operations": [
            {"flatten": "metadata", "rename": {}}  # Two keys!
        ]
    }

    plugin = RowDataReshape(config)
    row = {"id": 1}

    with pytest.raises(ValueError) as exc_info:
        plugin.transform(row, context={})

    error_msg = str(exc_info.value)
    assert "exactly one key" in error_msg.lower()


def test_row_data_reshape_validates_unknown_operation():
    """RowDataReshape should reject unknown operations."""
    config = {
        "operations": [
            {"unknown_op": "value"}
        ]
    }

    plugin = RowDataReshape(config)
    row = {"id": 1}

    with pytest.raises(ValueError) as exc_info:
        plugin.transform(row, context={})

    error_msg = str(exc_info.value)
    assert "unknown operation" in error_msg.lower()


def test_row_data_reshape_provides_context_on_failure():
    """RowDataReshape should provide helpful context on operation failure."""
    config = {
        "operations": [
            {"cast": {"field": "score", "type": "float"}}
        ]
    }

    plugin = RowDataReshape(config)
    row = {"id": 1, "score": "invalid"}  # Can't cast to float

    with pytest.raises(ValueError) as exc_info:
        plugin.transform(row, context={})

    error_msg = str(exc_info.value)
    assert "cast" in error_msg.lower()
    assert "invalid" in error_msg


def test_row_data_reshape_complex_pipeline():
    """Test realistic multi-operation pipeline."""
    config = {
        "operations": [
            # 1. Flatten nested metadata
            {"flatten": "metadata"},

            # 2. Rename flattened fields
            {"rename": {
                "metadata.confidence": "confidence",
                "metadata.sample_size": "n",
                "metadata.version": "v"
            }},

            # 3. Cast string to float
            {"cast": {"field": "confidence", "type": "float"}},

            # 4. Extract nested score
            {"extract": {"parent": "analysis", "child": "score"}},

            # 5. Remove unwanted fields
            {"exclude_fields": ["v", "prompt", "response"]},

            # 6. Keep only final fields
            {"filter_fields": ["id", "score", "confidence", "n"]}
        ]
    }

    plugin = RowDataReshape(config)

    row = {
        "id": 1,
        "prompt": "What is 2+2?",
        "response": "4",
        "analysis": {
            "score": 0.95,
            "reasoning": "correct"
        },
        "metadata": {
            "confidence": "0.98",
            "sample_size": 100,
            "version": "1.0"
        }
    }

    result = plugin.transform(row, context={})

    expected = {
        "id": 1,
        "score": 0.95,
        "confidence": 0.98,
        "n": 100
    }

    assert result == expected
    assert isinstance(result["confidence"], float)


def test_row_data_reshape_validates_flatten_type():
    """RowDataReshape should reject flatten with wrong argument type."""
    config = {"operations": [{"flatten": {"wrong": "type"}}]}
    plugin = RowDataReshape(config)

    with pytest.raises(ValueError) as exc_info:
        plugin.transform({"id": 1}, context={})

    error_msg = str(exc_info.value)
    assert "flatten expects string" in error_msg.lower()


def test_row_data_reshape_validates_rename_type():
    """RowDataReshape should reject rename with wrong argument type."""
    config = {"operations": [{"rename": ["wrong"]}]}
    plugin = RowDataReshape(config)

    with pytest.raises(ValueError) as exc_info:
        plugin.transform({"id": 1}, context={})

    error_msg = str(exc_info.value)
    assert "rename expects dict" in error_msg.lower()


def test_row_data_reshape_validates_filter_fields_type():
    """RowDataReshape should reject filter_fields with wrong argument type."""
    config = {"operations": [{"filter_fields": "wrong"}]}
    plugin = RowDataReshape(config)

    with pytest.raises(ValueError) as exc_info:
        plugin.transform({"id": 1}, context={})

    error_msg = str(exc_info.value)
    assert "filter_fields expects list" in error_msg.lower()


def test_row_data_reshape_validates_exclude_fields_type():
    """RowDataReshape should reject exclude_fields with wrong argument type."""
    config = {"operations": [{"exclude_fields": "wrong"}]}
    plugin = RowDataReshape(config)

    with pytest.raises(ValueError) as exc_info:
        plugin.transform({"id": 1}, context={})

    error_msg = str(exc_info.value)
    assert "exclude_fields expects list" in error_msg.lower()


def test_row_data_reshape_validates_extract_type():
    """RowDataReshape should reject extract with wrong argument type."""
    config = {"operations": [{"extract": "wrong"}]}
    plugin = RowDataReshape(config)

    with pytest.raises(ValueError) as exc_info:
        plugin.transform({"id": 1}, context={})

    error_msg = str(exc_info.value)
    assert "extract expects dict" in error_msg.lower()


def test_row_data_reshape_validates_extract_missing_parent():
    """RowDataReshape should reject extract missing parent field."""
    config = {"operations": [{"extract": {"child": "field"}}]}
    plugin = RowDataReshape(config)

    with pytest.raises(ValueError) as exc_info:
        plugin.transform({"id": 1}, context={})

    error_msg = str(exc_info.value)
    assert "requires both 'parent' and 'child'" in error_msg.lower()


def test_row_data_reshape_validates_extract_missing_child():
    """RowDataReshape should reject extract missing child field."""
    config = {"operations": [{"extract": {"parent": "field"}}]}
    plugin = RowDataReshape(config)

    with pytest.raises(ValueError) as exc_info:
        plugin.transform({"id": 1}, context={})

    error_msg = str(exc_info.value)
    assert "requires both 'parent' and 'child'" in error_msg.lower()


def test_row_data_reshape_validates_cast_type():
    """RowDataReshape should reject cast with wrong argument type."""
    config = {"operations": [{"cast": "wrong"}]}
    plugin = RowDataReshape(config)

    with pytest.raises(ValueError) as exc_info:
        plugin.transform({"id": 1}, context={})

    error_msg = str(exc_info.value)
    assert "cast expects dict" in error_msg.lower()


def test_row_data_reshape_validates_cast_missing_field():
    """RowDataReshape should reject cast missing field."""
    config = {"operations": [{"cast": {"type": "float"}}]}
    plugin = RowDataReshape(config)

    with pytest.raises(ValueError) as exc_info:
        plugin.transform({"id": 1}, context={})

    error_msg = str(exc_info.value)
    assert "requires both 'field' and 'type'" in error_msg.lower()


def test_row_data_reshape_validates_cast_missing_type():
    """RowDataReshape should reject cast missing type."""
    config = {"operations": [{"cast": {"field": "score"}}]}
    plugin = RowDataReshape(config)

    with pytest.raises(ValueError) as exc_info:
        plugin.transform({"id": 1}, context={})

    error_msg = str(exc_info.value)
    assert "requires both 'field' and 'type'" in error_msg.lower()
