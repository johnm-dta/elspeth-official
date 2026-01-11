"""Edge case tests for validation system."""

from elspeth.core.validation import validate_schema


class TestNumericEdgeCases:
    """Edge cases for numeric validation."""

    def test_nan_is_valid_number(self):
        """NaN should be considered a valid number."""
        schema = {"type": "number"}
        data = float('nan')
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0, "NaN should be a valid number"

    def test_infinity_is_valid_number(self):
        """Positive infinity should be a valid number."""
        schema = {"type": "number"}
        data = float('inf')
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0, "Infinity should be a valid number"

    def test_negative_infinity_is_valid_number(self):
        """Negative infinity should be a valid number."""
        schema = {"type": "number"}
        data = float('-inf')
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0, "-Infinity should be a valid number"

    def test_zero_is_valid_number(self):
        """Zero should be a valid number."""
        schema = {"type": "number"}
        data = 0
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_negative_zero_is_valid_number(self):
        """Negative zero should be a valid number."""
        schema = {"type": "number"}
        data = -0.0
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_max_int(self):
        """Very large integer should be valid."""
        schema = {"type": "integer"}
        data = 2**63 - 1  # Max 64-bit signed int
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_min_int(self):
        """Very small (negative) integer should be valid."""
        schema = {"type": "integer"}
        data = -(2**63)
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_floating_point_precision(self):
        """Floating point values should work despite precision issues."""
        schema = {"type": "number"}
        # Classic floating point issue: 0.1 + 0.2 != 0.3
        data = 0.1 + 0.2  # Actually 0.30000000000000004
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0


class TestStringEdgeCases:
    """Edge cases for string validation."""

    def test_empty_string_is_valid(self):
        """Empty string should be a valid string."""
        schema = {"type": "string"}
        data = ""
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_whitespace_only_is_valid_string(self):
        """Whitespace-only string should be valid."""
        schema = {"type": "string"}
        data = "   \t\n  "
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_unicode_nfc_is_valid(self):
        """Unicode NFC normalized string should be valid."""
        schema = {"type": "string"}
        data = "café"  # NFC form (single codepoint for é)
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_unicode_nfd_is_valid(self):
        """Unicode NFD normalized string should be valid."""
        schema = {"type": "string"}
        # NFD form (e + combining accent)
        import unicodedata
        data = unicodedata.normalize('NFD', "café")
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_nfc_and_nfd_are_different_strings(self):
        """NFC and NFD forms are different for validation purposes."""
        import unicodedata
        nfc = "café"
        nfd = unicodedata.normalize('NFD', "café")
        # They're different strings (different byte sequences)
        assert nfc != nfd
        # But both are valid strings
        schema = {"type": "string"}
        assert list(validate_schema(nfc, schema)) == []
        assert list(validate_schema(nfd, schema)) == []

    def test_very_long_string(self):
        """Very long string should be valid."""
        schema = {"type": "string"}
        data = "a" * 100000
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0


class TestNullHandling:
    """Edge cases for null/None handling."""

    def test_null_vs_missing_field(self):
        """null value is different from missing field."""
        schema = {
            "type": "object",
            "required": ["field1"],
            "properties": {
                "field1": {"type": ["number", "null"]}  # Allows null
            }
        }

        # null value is valid (field exists)
        data_with_null = {"field1": None}
        errors = list(validate_schema(data_with_null, schema))
        assert len(errors) == 0

        # Missing key is invalid (required field absent)
        data_missing = {}
        errors = list(validate_schema(data_missing, schema))
        assert len(errors) > 0
        assert "required" in errors[0].message.lower()

    def test_null_not_allowed_by_default(self):
        """null should not be valid for number type."""
        schema = {"type": "number"}
        data = None
        errors = list(validate_schema(data, schema))
        assert len(errors) > 0


class TestArrayEdgeCases:
    """Edge cases for array validation."""

    def test_empty_array_is_valid(self):
        """Empty array should be valid."""
        schema = {"type": "array", "items": {"type": "number"}}
        data = []
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_single_element_array(self):
        """Single element array should be valid."""
        schema = {"type": "array", "items": {"type": "number"}}
        data = [42]
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_very_large_array(self):
        """Very large array should be valid."""
        schema = {"type": "array", "items": {"type": "number"}}
        data = list(range(10000))
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0


class TestObjectDepth:
    """Edge cases for nested object depth."""

    def test_deeply_nested_object(self):
        """Deeply nested object should be valid."""
        schema = {"type": "object"}

        # Create 100-level deep nested object
        data = {"level": 100}
        current = data
        for i in range(99, 0, -1):
            current["nested"] = {"level": i}
            current = current["nested"]

        errors = list(validate_schema(data, schema))
        assert len(errors) == 0


class TestCollectionEdgeCases:
    """Edge cases for collection type validation."""

    def test_empty_collection_is_valid(self):
        """Empty collection should be valid."""
        schema = {"type": "collection", "item_schema": {"type": "number"}}
        data = {}
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_single_element_collection(self):
        """Collection with single field containing one element should be valid."""
        schema = {"type": "collection", "item_schema": {"type": "number"}}
        data = {"score": [42]}
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_unequal_array_lengths_in_collection(self):
        """Collection can have arrays of different lengths."""
        schema = {"type": "collection", "item_schema": {"type": "number"}}
        data = {
            "scores": [1, 2, 3],
            "ratings": [4, 5]
        }
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0

    def test_nested_objects_in_collection(self):
        """Collection can contain arrays of nested objects."""
        schema = {
            "type": "collection",
            "item_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "number"}
                }
            }
        }
        data = {
            "items": [
                {"name": "first", "value": 1},
                {"name": "second", "value": 2}
            ]
        }
        errors = list(validate_schema(data, schema))
        assert len(errors) == 0


class TestErrorMessages:
    """Test quality of validation error messages."""

    def test_error_message_includes_path(self):
        """Error messages should include JSON pointer path."""
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"}
                    }
                }
            }
        }
        data = {"nested": {"value": "not_a_number"}}
        errors = list(validate_schema(data, schema))
        assert len(errors) > 0
        assert "nested.value" in errors[0].format()

    def test_error_message_includes_type(self):
        """Error messages should include expected type."""
        schema = {"type": "number"}
        data = "not_a_number"
        errors = list(validate_schema(data, schema))
        assert len(errors) > 0
        assert "number" in errors[0].message.lower()

    def test_error_message_for_array_index(self):
        """Error messages should include array index in path."""
        schema = {
            "type": "array",
            "items": {"type": "number"}
        }
        data = [1, 2, "not_a_number", 4]
        errors = list(validate_schema(data, schema))
        assert len(errors) > 0
        assert "[2]" in errors[0].format()

    def test_error_message_for_required_field(self):
        """Error messages should be clear for missing required fields."""
        schema = {
            "type": "object",
            "required": ["field1", "field2"],
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "number"}
            }
        }
        data = {"field1": "value"}  # field2 missing
        errors = list(validate_schema(data, schema))
        assert len(errors) > 0
        assert "field2" in errors[0].format()
        assert "required" in errors[0].message.lower()

    def test_error_message_for_enum_violation(self):
        """Error messages should show allowed enum values."""
        schema = {
            "type": "string",
            "enum": ["red", "green", "blue"]
        }
        data = "yellow"
        errors = list(validate_schema(data, schema))
        assert len(errors) > 0
        assert "one of" in errors[0].message.lower()
        # Should mention the allowed values
        error_text = errors[0].message
        assert "red" in error_text or "green" in error_text or "blue" in error_text

    def test_error_message_for_minimum_violation(self):
        """Error messages should show minimum constraint."""
        schema = {
            "type": "number",
            "minimum": 10
        }
        data = 5
        errors = list(validate_schema(data, schema))
        assert len(errors) > 0
        assert ">=" in errors[0].message
        assert "10" in errors[0].message

    def test_error_message_for_exclusive_minimum_violation(self):
        """Error messages should show exclusive minimum constraint."""
        schema = {
            "type": "number",
            "exclusiveMinimum": 10
        }
        data = 10
        errors = list(validate_schema(data, schema))
        assert len(errors) > 0
        assert ">" in errors[0].message
        assert "10" in errors[0].message
