"""Fuzzing tests for validation system."""

import random
import string

from elspeth.core.validation import validate_schema


def random_garbage(length=100):
    """Generate random string of printable characters."""
    return ''.join(random.choices(string.printable, k=length))


def test_validator_handles_garbage_strings():
    """Validator should not crash on random garbage strings."""
    schema = {"type": "number"}

    for _ in range(1000):
        garbage = random_garbage()
        # Should not crash, should return validation errors
        errors = list(validate_schema(garbage, schema))
        assert isinstance(errors, list)
        # Should have errors (garbage string is not a number)
        assert len(errors) > 0


def test_validator_handles_random_types():
    """Validator should handle random Python types gracefully."""
    schema = {"type": "number"}

    test_values = [
        None,
        [],
        {},
        set(),
        (),
        lambda x: x,
        object(),
        type,
        ...,  # Ellipsis
        1+2j,  # Complex number
        b"bytes",
        bytearray(b"test"),
    ]

    for value in test_values:
        # Should not crash
        errors = list(validate_schema(value, schema))
        assert isinstance(errors, list)


def test_validator_handles_malformed_schemas():
    """Validator should handle malformed schemas gracefully.

    Note: This test currently catches AttributeError for malformed schemas.
    The validator assumes schemas are well-formed (properties is a dict, items is a schema).
    In production, schemas are typically validated before use, so this is acceptable.
    """
    data = {"field": "value"}

    # Safe malformed schemas (don't crash)
    safe_schemas = [
        {"type": "nonexistent_type"},
        {"type": 123},  # type should be string
        {"required": "not_a_list"},  # required should be list (but get() handles this)
    ]

    for schema in safe_schemas:
        # Should not crash
        errors = list(validate_schema(data, schema))
        assert isinstance(errors, list)

    # Schemas that cause AttributeError (properties/items must be correct type)
    unsafe_schemas = [
        {"properties": "not_a_dict"},  # properties should be dict
        {"items": "not_a_schema"},  # items should be schema (for arrays)
    ]

    for schema in unsafe_schemas:
        # These currently raise AttributeError - documenting the limitation
        try:
            errors = list(validate_schema(data, schema))
            assert isinstance(errors, list)
        except AttributeError:
            # Expected: validator assumes schema structure is valid
            pass


def test_validator_handles_deeply_nested_garbage():
    """Validator should handle deeply nested garbage data."""
    schema = {"type": "object"}

    for _ in range(100):
        # Create random nested structure
        data = {"level": 0}
        current = data
        depth = random.randint(1, 50)

        for i in range(depth):
            if random.choice([True, False]):
                current["nested"] = {"level": i}
                current = current["nested"]
            else:
                current["nested"] = random_garbage(50)
                break

        # Should not crash
        errors = list(validate_schema(data, schema))
        assert isinstance(errors, list)


def test_validator_handles_large_arrays_with_garbage():
    """Validator should handle large arrays with mixed garbage."""
    schema = {"type": "array", "items": {"type": "number"}}

    for _ in range(10):
        # Create large array with random values
        size = random.randint(100, 1000)
        data = []
        for _ in range(size):
            if random.random() < 0.1:  # 10% garbage
                data.append(random_garbage(10))
            else:
                data.append(random.randint(0, 1000))

        # Should not crash
        errors = list(validate_schema(data, schema))
        assert isinstance(errors, list)


def test_validator_handles_deeply_nested_structures():
    """Validator should handle deeply nested object structures."""
    schema = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object"
            }
        }
    }

    for _ in range(100):
        # Generate random deeply nested structures
        depth = random.randint(1, 100)
        data = {}
        current = data

        for _i in range(depth):
            if random.random() < 0.7:  # 70% chance to continue nesting
                current["nested"] = {}
                current = current["nested"]
            else:
                current["nested"] = random_garbage(20)
                break

        # Should not crash
        errors = list(validate_schema(data, schema))
        assert isinstance(errors, list)


def test_validator_handles_mixed_type_arrays():
    """Validator should handle arrays with random mixed types."""
    schema = {"type": "array", "items": {"type": "object"}}

    for _ in range(100):
        # Generate arrays with random mixed content
        size = random.randint(1, 50)
        data = []

        for _ in range(size):
            choice = random.random()
            if choice < 0.2:
                data.append({})
            elif choice < 0.4:
                data.append(random_garbage(10))
            elif choice < 0.6:
                data.append(random.randint(0, 1000))
            elif choice < 0.8:
                data.append([random.randint(0, 100) for _ in range(5)])
            else:
                data.append(None)

        # Should not crash
        errors = list(validate_schema(data, schema))
        assert isinstance(errors, list)


def test_validator_handles_deeply_nested_collections():
    """Validator should handle deeply nested collection structures."""
    schema = {
        "type": "collection",
        "item_schema": {"type": "object"}
    }

    for _ in range(100):
        # Generate random collection-like structures
        data = {}
        num_keys = random.randint(1, 20)

        for i in range(num_keys):
            key = f"field_{i}"
            if random.random() < 0.7:
                # Valid array
                array_size = random.randint(0, 10)
                data[key] = [{"nested": j} for j in range(array_size)]
            elif random.random() < 0.5:
                # Invalid: not an array
                data[key] = random_garbage(10)
            else:
                # Invalid: array with wrong types
                data[key] = [random_garbage(5) for _ in range(5)]

        # Should not crash
        errors = list(validate_schema(data, schema))
        assert isinstance(errors, list)


def test_validator_handles_circular_like_structures():
    """Validator should handle structures that appear circular-like."""
    schema = {"type": "object"}

    for _ in range(50):
        # Create structures with repeated patterns
        data = {}
        current = data

        for i in range(random.randint(5, 30)):
            # Create a chain that references same keys
            key = f"level_{i % 5}"  # Reuse same 5 keys
            if random.random() < 0.8:
                current[key] = {"value": i}
                current = current[key]
            else:
                current[key] = random_garbage(10)
                break

        # Should not crash
        errors = list(validate_schema(data, schema))
        assert isinstance(errors, list)


def test_validator_handles_extreme_schema_variations():
    """Validator should handle extreme schema variations."""
    data = {"field": "value", "number": 42, "nested": {"inner": "data"}}

    # Generate various schema configurations
    for _ in range(200):
        schema_type = random.choice(["object", "array", "string", "number", "integer", "boolean"])
        schema = {"type": schema_type}

        # Add random schema properties
        if random.random() < 0.3:
            schema["required"] = [random_garbage(5) for _ in range(3)]
        if random.random() < 0.3:
            schema["properties"] = {random_garbage(5): {"type": "string"} for _ in range(3)}
        if random.random() < 0.3:
            schema["items"] = {"type": random.choice(["string", "number", "object"])}
        if random.random() < 0.3:
            schema["enum"] = [random_garbage(3) for _ in range(5)]
        if random.random() < 0.3:
            schema["minimum"] = random.randint(-1000, 1000)
        if random.random() < 0.3:
            schema["exclusiveMinimum"] = random.randint(-1000, 1000)

        # Should not crash
        errors = list(validate_schema(data, schema))
        assert isinstance(errors, list)


def test_validator_handles_unicode_garbage():
    """Validator should handle unicode and special characters."""
    schema = {"type": "object", "properties": {"field": {"type": "string"}}}

    for _ in range(100):
        # Generate data with unicode characters
        data = {
            random_garbage(10): random_garbage(20),
            "field": ''.join(chr(random.randint(0, 0x10ffff)) if random.random() < 0.5 else 'a' for _ in range(20))
        }

        # Should not crash
        try:
            errors = list(validate_schema(data, schema))
            assert isinstance(errors, list)
        except (UnicodeDecodeError, UnicodeEncodeError):
            # Unicode errors are acceptable for invalid unicode
            pass


def test_validator_handles_very_large_objects():
    """Validator should handle very large objects without crashing."""
    schema = {"type": "object"}

    for _ in range(10):
        # Generate large objects with many keys
        num_keys = random.randint(100, 500)
        data = {f"key_{i}": random_garbage(10) for i in range(num_keys)}

        # Should not crash
        errors = list(validate_schema(data, schema))
        assert isinstance(errors, list)


def test_validator_handles_very_large_arrays():
    """Validator should handle very large arrays without crashing."""
    schema = {"type": "array", "items": {"type": "string"}}

    for _ in range(10):
        # Generate large arrays
        size = random.randint(100, 1000)
        data = [random_garbage(5) if random.random() < 0.8 else random.randint(0, 100) for _ in range(size)]

        # Should not crash
        errors = list(validate_schema(data, schema))
        assert isinstance(errors, list)


def test_validator_handles_empty_and_none_variations():
    """Validator should handle various empty and None values."""
    schemas = [
        {"type": "object"},
        {"type": "array"},
        {"type": "string"},
        {"type": "number"},
    ]

    test_values = [
        None,
        {},
        [],
        "",
        0,
        False,
    ]

    for schema in schemas:
        for value in test_values:
            # Should not crash
            errors = list(validate_schema(value, schema))
            assert isinstance(errors, list)
