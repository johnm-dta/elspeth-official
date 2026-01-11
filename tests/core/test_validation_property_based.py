"""Property-based tests for validation system using hypothesis."""

from hypothesis import given
from hypothesis import strategies as st

from elspeth.core.validation import validate_schema


@given(st.integers())
def test_validator_accepts_any_integer(value):
    """Validator should accept any integer for integer schema."""
    schema = {"type": "integer"}
    errors = list(validate_schema(value, schema))
    assert len(errors) == 0


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_validator_accepts_any_finite_number(value):
    """Validator should accept any finite number for number schema."""
    schema = {"type": "number"}
    errors = list(validate_schema(value, schema))
    assert len(errors) == 0


@given(st.text())
def test_validator_accepts_any_string(value):
    """Validator should accept any string for string schema."""
    schema = {"type": "string"}
    errors = list(validate_schema(value, schema))
    assert len(errors) == 0


@given(st.booleans())
def test_validator_accepts_any_boolean(value):
    """Validator should accept any boolean for boolean schema."""
    schema = {"type": "boolean"}
    errors = list(validate_schema(value, schema))
    assert len(errors) == 0


@given(st.dictionaries(st.text(), st.integers()))
def test_validator_accepts_valid_objects(value):
    """Validator should accept dict with integer values for object schema."""
    schema = {
        "type": "object",
        "additionalProperties": {"type": "integer"}
    }
    # Note: validate_schema expects Mapping, dict is Mapping
    errors = list(validate_schema(value, schema))
    assert len(errors) == 0


@given(st.lists(st.integers()))
def test_validator_accepts_valid_arrays(value):
    """Validator should accept list of integers for array schema."""
    schema = {
        "type": "array",
        "items": {"type": "integer"}
    }
    errors = list(validate_schema(value, schema))
    assert len(errors) == 0


@given(st.text())
def test_validator_rejects_wrong_types(value):
    """Validator should reject string when number expected."""
    schema = {"type": "number"}
    errors = list(validate_schema(value, schema))
    assert len(errors) > 0  # Should have errors


@given(st.dictionaries(
    st.text(min_size=1),
    st.lists(st.integers(), min_size=1)
))
def test_validator_accepts_valid_collections(value):
    """Validator should accept collection with integer arrays."""
    schema = {
        "type": "collection",
        "item_schema": {
            "type": "integer"
        }
    }
    # Collection has structure: {"field1": [int, int, ...], "field2": [int, ...]}
    errors = list(validate_schema(value, schema))
    assert len(errors) == 0
