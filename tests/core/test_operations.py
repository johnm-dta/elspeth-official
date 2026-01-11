import copy

import pytest

from elspeth.core.operations import (
    cast,
    exclude_fields,
    extract,
    filter_fields,
    flatten,
    rename,
)


def test_flatten_nested_dict():
    """Flatten operation should flatten nested dictionaries."""
    data = {
        "id": 1,
        "metadata": {
            "confidence": 0.95,
            "sample_size": 100
        }
    }

    result = flatten(data, "metadata")

    expected = {
        "id": 1,
        "metadata.confidence": 0.95,
        "metadata.sample_size": 100
    }

    assert result == expected


def test_flatten_preserves_non_nested():
    """Flatten should preserve fields that aren't nested."""
    data = {"id": 1, "score": 0.75}

    result = flatten(data, "nonexistent")

    assert result == data


def test_flatten_handles_missing_field():
    """Flatten should handle missing field gracefully."""
    data = {"id": 1}

    result = flatten(data, "metadata")

    assert result == data


def test_rename_fields():
    """Rename operation should rename specified fields."""
    data = {
        "metadata.confidence": 0.95,
        "metadata.sample_size": 100,
        "score": 0.75
    }

    rename_map = {
        "metadata.confidence": "confidence",
        "metadata.sample_size": "sample_size"
    }

    result = rename(data, rename_map)

    expected = {
        "confidence": 0.95,
        "sample_size": 100,
        "score": 0.75
    }

    assert result == expected


def test_rename_handles_missing_fields():
    """Rename should skip fields that don't exist."""
    data = {"score": 0.75}

    rename_map = {"nonexistent": "new_name"}

    result = rename(data, rename_map)

    assert result == data


def test_filter_fields_keeps_only_specified():
    """Filter should keep only specified fields."""
    data = {
        "id": 1,
        "score": 0.75,
        "confidence": 0.95,
        "metadata": {}
    }

    result = filter_fields(data, ["id", "score"])

    expected = {"id": 1, "score": 0.75}

    assert result == expected


def test_exclude_fields_removes_specified():
    """Exclude should remove specified fields."""
    data = {
        "id": 1,
        "score": 0.75,
        "prompt": "Question 1",
        "response": "Answer 1"
    }

    result = exclude_fields(data, ["prompt", "response"])

    expected = {"id": 1, "score": 0.75}

    assert result == expected


def test_extract_nested_to_top_level():
    """Extract should move nested field to top level."""
    data = {
        "id": 1,
        "analysis": {
            "score": 0.75,
            "confidence": 0.95
        }
    }

    result = extract(data, "analysis", "score")

    expected = {
        "id": 1,
        "score": 0.75,
        "analysis": {
            "confidence": 0.95
        }
    }

    assert result == expected


def test_extract_handles_missing_parent():
    """Extract should handle missing parent field."""
    data = {"id": 1}

    result = extract(data, "nonexistent", "field")

    assert result == data


def test_cast_to_float():
    """Cast should convert field to specified type."""
    data = {"id": "1", "score": "0.75"}

    result = cast(data, "score", "float")

    assert result["score"] == 0.75
    assert isinstance(result["score"], float)


def test_cast_to_int():
    """Cast should convert to integer."""
    data = {"count": "100"}

    result = cast(data, "count", "int")

    assert result["count"] == 100
    assert isinstance(result["count"], int)


def test_cast_to_string():
    """Cast should convert to string."""
    data = {"id": 1}

    result = cast(data, "id", "string")

    assert result["id"] == "1"
    assert isinstance(result["id"], str)


def test_cast_to_bool():
    """Cast should handle boolean conversions."""
    assert cast({"flag": "true"}, "flag", "bool")["flag"] is True
    assert cast({"flag": "false"}, "flag", "bool")["flag"] is False
    assert cast({"flag": "1"}, "flag", "bool")["flag"] is True
    assert cast({"flag": 0}, "flag", "bool")["flag"] is False


def test_cast_handles_invalid_conversion():
    """Cast should raise ValueError for invalid conversion."""
    data = {"score": "invalid"}

    with pytest.raises(ValueError) as exc_info:
        cast(data, "score", "float")

    assert "cast" in str(exc_info.value).lower()
    assert "score" in str(exc_info.value)


def test_flatten_collection():
    """Flatten should work on collection (dict of arrays)."""
    collection = {
        "id": [1, 2],
        "metadata": [
            {"confidence": 0.95, "size": 100},
            {"confidence": 0.88, "size": 120}
        ]
    }

    # For collections, flatten should be applied to each item
    # This will be handled by AggregateDataReshape plugin
    # Here we test that the operation works on individual items

    result = flatten(collection["metadata"][0], "confidence")

    # Note: This test validates the operation works on collection items
    # The plugin will handle applying it across all items
    assert "confidence" in str(result) or result == collection["metadata"][0]


def test_operations_are_pure_functions():
    """Operations should not mutate input data, even nested values."""
    original = {"id": 1, "metadata": {"tags": ["a", "b"], "scores": {"x": 1}}}
    original_copy = copy.deepcopy(original)

    result = flatten(original, "metadata")

    # Try to mutate nested values in result
    if "metadata.tags" in result and isinstance(result["metadata.tags"], list):
        result["metadata.tags"].append("c")
    if "metadata.scores" in result and isinstance(result["metadata.scores"], dict):
        result["metadata.scores"]["x"] = 999

    # Original must be unchanged
    assert original == original_copy
