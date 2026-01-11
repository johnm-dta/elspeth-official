"""Compare our validator against python-jsonschema reference implementation."""

import importlib.util

import pytest

from elspeth.core.validation import validate_schema as our_validator

JSONSCHEMA_AVAILABLE = importlib.util.find_spec("jsonschema") is not None

if JSONSCHEMA_AVAILABLE:
    from jsonschema import SchemaError as RefSchemaError
    from jsonschema import ValidationError as RefValidationError
    from jsonschema import validate as ref_validator


@pytest.mark.skipif(not JSONSCHEMA_AVAILABLE, reason="jsonschema not installed")
class TestReferenceComparison:
    """Compare our validator against jsonschema reference."""

    def test_both_accept_valid_object(self):
        """Both validators should accept valid object."""
        schema = {"type": "object", "required": ["name", "age"], "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
        data = {"name": "Alice", "age": 30}

        # Our validator
        our_errors = list(our_validator(data, schema))
        assert len(our_errors) == 0

        # Reference validator
        try:
            ref_validator(data, schema)
            ref_valid = True
        except RefValidationError:
            ref_valid = False

        assert ref_valid, "Reference should also accept"

    def test_both_reject_invalid_type(self):
        """Both validators should reject wrong type."""
        schema = {"type": "number"}
        data = "not_a_number"

        # Our validator
        our_errors = list(our_validator(data, schema))
        assert len(our_errors) > 0

        # Reference validator
        with pytest.raises(RefValidationError):
            ref_validator(data, schema)

    def test_both_reject_missing_required(self):
        """Both validators should reject missing required field."""
        schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
        data = {}

        # Our validator
        our_errors = list(our_validator(data, schema))
        assert len(our_errors) > 0
        assert "required" in our_errors[0].message.lower()

        # Reference validator
        with pytest.raises(RefValidationError) as exc_info:
            ref_validator(data, schema)
        assert "required" in str(exc_info.value).lower()

    def test_both_handle_nested_objects(self):
        """Both validators should handle nested objects."""
        schema = {
            "type": "object",
            "properties": {"person": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}},
        }

        valid_data = {"person": {"name": "Bob", "age": 25}}
        invalid_data = {"person": {"age": 25}}  # Missing name

        # Valid - both should accept
        assert len(list(our_validator(valid_data, schema))) == 0
        ref_validator(valid_data, schema)  # Should not raise

        # Invalid - both should reject
        assert len(list(our_validator(invalid_data, schema))) > 0
        with pytest.raises(RefValidationError):
            ref_validator(invalid_data, schema)

    def test_both_handle_arrays(self):
        """Both validators should handle arrays."""
        schema = {"type": "array", "items": {"type": "number"}}

        valid_data = [1, 2, 3]
        invalid_data = [1, "two", 3]

        # Valid
        assert len(list(our_validator(valid_data, schema))) == 0
        ref_validator(valid_data, schema)

        # Invalid
        assert len(list(our_validator(invalid_data, schema))) > 0
        with pytest.raises(RefValidationError):
            ref_validator(invalid_data, schema)

    def test_both_handle_string_type(self):
        """Both validators should handle string type."""
        schema = {"type": "string"}

        # Valid
        assert len(list(our_validator("hello", schema))) == 0
        ref_validator("hello", schema)

        # Invalid
        assert len(list(our_validator(123, schema))) > 0
        with pytest.raises(RefValidationError):
            ref_validator(123, schema)

    def test_both_handle_boolean_type(self):
        """Both validators should handle boolean type."""
        schema = {"type": "boolean"}

        # Valid
        assert len(list(our_validator(True, schema))) == 0
        ref_validator(True, schema)

        assert len(list(our_validator(False, schema))) == 0
        ref_validator(False, schema)

        # Invalid
        assert len(list(our_validator("true", schema))) > 0
        with pytest.raises(RefValidationError):
            ref_validator("true", schema)

    def test_both_handle_integer_type(self):
        """Both validators should handle integer type."""
        schema = {"type": "integer"}

        # Valid
        assert len(list(our_validator(42, schema))) == 0
        ref_validator(42, schema)

        # Invalid - float
        assert len(list(our_validator(42.5, schema))) > 0
        with pytest.raises(RefValidationError):
            ref_validator(42.5, schema)

    def test_null_type_not_implemented(self):
        """Our validator does not implement null type (known difference)."""
        schema = {"type": "null"}

        # Reference validator accepts null
        ref_validator(None, schema)

        # Our validator treats None as missing value
        our_errors = list(our_validator(None, schema))
        assert len(our_errors) > 0
        assert "missing" in our_errors[0].message.lower()

        # This is a known difference - null type not implemented in our validator

    def test_both_handle_empty_object(self):
        """Both validators should handle empty object."""
        schema = {"type": "object"}

        # Valid
        assert len(list(our_validator({}, schema))) == 0
        ref_validator({}, schema)

    def test_both_handle_empty_array(self):
        """Both validators should handle empty array."""
        schema = {"type": "array"}

        # Valid
        assert len(list(our_validator([], schema))) == 0
        ref_validator([], schema)

    def test_both_handle_empty_string(self):
        """Both validators should handle empty string."""
        schema = {"type": "string"}

        # Valid
        assert len(list(our_validator("", schema))) == 0
        ref_validator("", schema)

    def test_additional_properties_not_implemented(self):
        """Our validator does not implement additionalProperties (known difference)."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": False}

        # Valid - only allowed properties - both should accept
        valid_data = {"name": "Alice"}
        assert len(list(our_validator(valid_data, schema))) == 0
        ref_validator(valid_data, schema)

        # Invalid - extra properties
        invalid_data = {"name": "Alice", "age": 30}

        # Reference validator rejects extra properties
        with pytest.raises(RefValidationError):
            ref_validator(invalid_data, schema)

        # Our validator does not enforce additionalProperties: False (known difference)
        our_errors = list(our_validator(invalid_data, schema))
        assert len(our_errors) == 0  # We don't reject extra properties

    def test_collection_type_difference(self):
        """Our collection type is NOT in JSON Schema spec."""
        schema = {"type": "collection", "item_schema": {"type": "number"}}
        data = {"scores": [1.0, 2.0, 3.0], "ratings": [4.5, 5.0]}

        # Our validator should handle it
        our_errors = list(our_validator(data, schema))
        assert len(our_errors) == 0

        # Reference validator will reject unknown type (schema error, not validation error)
        with pytest.raises((RefValidationError, RefSchemaError)):
            ref_validator(data, schema)

        # This is expected - collection is our extension


@pytest.mark.skipif(not JSONSCHEMA_AVAILABLE, reason="jsonschema not installed")
class TestEdgeCaseComparison:
    """Compare edge case handling."""

    def test_both_handle_nan(self):
        """Both validators should handle NaN as valid number."""
        import math

        schema = {"type": "number"}
        data = math.nan

        # Our validator - should accept NaN
        our_errors = list(our_validator(data, schema))
        assert len(our_errors) == 0

        # Reference validator - should also accept NaN
        ref_validator(data, schema)  # Should not raise

    def test_both_handle_infinity(self):
        """Both validators should handle infinity as valid number."""
        import math

        schema = {"type": "number"}

        # Positive infinity
        assert len(list(our_validator(math.inf, schema))) == 0
        ref_validator(math.inf, schema)

        # Negative infinity
        assert len(list(our_validator(-math.inf, schema))) == 0
        ref_validator(-math.inf, schema)

    def test_both_handle_unicode_strings(self):
        """Both validators should handle unicode strings."""
        schema = {"type": "string"}
        data = "Hello 世界 🌍"

        # Our validator
        assert len(list(our_validator(data, schema))) == 0

        # Reference validator
        ref_validator(data, schema)

    def test_both_handle_deeply_nested(self):
        """Both validators should handle deeply nested structures."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {"level2": {"type": "object", "properties": {"level3": {"type": "object", "properties": {"value": {"type": "number"}}}}}},
                }
            },
        }

        valid_data = {"level1": {"level2": {"level3": {"value": 42}}}}
        invalid_data = {"level1": {"level2": {"level3": {"value": "not_a_number"}}}}

        # Valid
        assert len(list(our_validator(valid_data, schema))) == 0
        ref_validator(valid_data, schema)

        # Invalid
        assert len(list(our_validator(invalid_data, schema))) > 0
        with pytest.raises(RefValidationError):
            ref_validator(invalid_data, schema)

    def test_both_handle_mixed_array(self):
        """Both validators should handle arrays with different types when no items schema."""
        schema = {"type": "array"}

        # Mixed types array - should be valid without items constraint
        data = [1, "two", 3.0, True, None]
        assert len(list(our_validator(data, schema))) == 0
        ref_validator(data, schema)
