"""Tests for validation system."""

from elspeth.core.validation import validate_schema


def test_collection_type_valid():
    """Collection type should validate correctly."""
    schema = {
        "type": "collection",
        "item_schema": {
            "type": "number"
        }
    }

    # Valid collection: object with array values (columnar data)
    data = {
        "score": [0.75, 0.82, 0.91],
        "confidence": [0.8, 0.9, 0.85]
    }

    errors = list(validate_schema(data, schema))
    assert len(errors) == 0, f"Expected no errors, got: {errors}"


def test_collection_type_invalid_not_object():
    """Collection must be an object (dict)."""
    schema = {
        "type": "collection",
        "item_schema": {"type": "object"}
    }

    # Invalid: collection is not an object
    data = [{"score": 0.75}, {"score": 0.82}]

    errors = list(validate_schema(data, schema))
    assert len(errors) > 0
    assert "must be of type collection" in errors[0].message.lower()


def test_collection_type_validates_array_elements():
    """Collection should validate all array elements against item_schema."""
    schema = {
        "type": "collection",
        "item_schema": {
            "type": "number"
        }
    }

    # Invalid: second element in score array is a string
    data = {
        "score": [0.75, "invalid", 0.91]
    }

    errors = list(validate_schema(data, schema))
    assert len(errors) > 0
    # Check that the error is about the invalid element
    assert "score[1]" in errors[0].format()


def test_collection_empty():
    """Empty collection is valid."""
    schema = {
        "type": "collection",
        "item_schema": {"type": "object"}
    }
    data = {}
    errors = list(validate_schema(data, schema))
    assert len(errors) == 0


def test_collection_mixed_array_lengths():
    """Collection can have arrays of different lengths."""
    schema = {
        "type": "collection",
        "item_schema": {
            "type": "number"
        }
    }

    # Different length arrays - this is valid for collection
    data = {
        "score": [0.75, 0.82],
        "confidence": [0.8, 0.9, 0.95]  # Different length OK
    }

    errors = list(validate_schema(data, schema))
    assert len(errors) == 0


def test_collection_field_not_array():
    """Collection fields must be arrays."""
    schema = {
        "type": "collection",
        "item_schema": {"type": "number"}
    }

    # Invalid: field contains scalar instead of array
    data = {
        "score": [0.75, 0.82],
        "confidence": 0.9  # Should be array!
    }

    errors = list(validate_schema(data, schema))
    assert len(errors) > 0
    assert "collection field must be an array" in errors[0].message
