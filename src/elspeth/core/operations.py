"""Shared data manipulation operations for reshape plugins."""

from __future__ import annotations

import copy
import json
from typing import Any


def flatten(data: dict[str, Any], field: str, separator: str = ".") -> dict[str, Any]:
    """
    Flatten nested dictionary field to top level with dotted keys.

    Args:
        data: Input dictionary
        field: Field to flatten (must be a dict)
        separator: Separator for flattened keys (default: ".")

    Returns:
        Dictionary with flattened fields

    Example:
        >>> flatten({"a": 1, "b": {"c": 2, "d": 3}}, "b")
        {"a": 1, "b.c": 2, "b.d": 3}
    """
    if field not in data:
        return copy.deepcopy(data)

    nested = data[field]
    if not isinstance(nested, dict):
        return copy.deepcopy(data)

    result = {k: copy.deepcopy(v) for k, v in data.items() if k != field}

    for nested_key, nested_value in nested.items():
        flattened_key = f"{field}{separator}{nested_key}"
        result[flattened_key] = copy.deepcopy(nested_value)

    return result


def rename(data: dict[str, Any], rename_map: dict[str, str]) -> dict[str, Any]:
    """
    Rename fields according to mapping.

    Args:
        data: Input dictionary
        rename_map: Mapping of old_name -> new_name

    Returns:
        Dictionary with renamed fields

    Example:
        >>> rename({"old": 1, "keep": 2}, {"old": "new"})
        {"new": 1, "keep": 2}
    """
    result = {}

    for key, value in data.items():
        new_key = rename_map.get(key, key)
        result[new_key] = copy.deepcopy(value)

    return result


def filter_fields(data: dict[str, Any], keep_fields: list[str]) -> dict[str, Any]:
    """
    Keep only specified fields.

    Args:
        data: Input dictionary
        keep_fields: List of fields to keep

    Returns:
        Dictionary with only specified fields

    Example:
        >>> filter_fields({"a": 1, "b": 2, "c": 3}, ["a", "c"])
        {"a": 1, "c": 3}
    """
    keep_set = set(keep_fields)
    return {k: copy.deepcopy(v) for k, v in data.items() if k in keep_set}


def exclude_fields(data: dict[str, Any], exclude: list[str]) -> dict[str, Any]:
    """
    Remove specified fields.

    Args:
        data: Input dictionary
        exclude: List of fields to remove

    Returns:
        Dictionary without excluded fields

    Example:
        >>> exclude_fields({"a": 1, "b": 2, "c": 3}, ["b"])
        {"a": 1, "c": 3}
    """
    exclude_set = set(exclude)
    return {k: copy.deepcopy(v) for k, v in data.items() if k not in exclude_set}


def extract(data: dict[str, Any], parent_field: str, child_field: str) -> dict[str, Any]:
    """
    Extract nested field to top level and remove from parent.

    Args:
        data: Input dictionary
        parent_field: Parent dictionary field
        child_field: Field to extract from parent

    Returns:
        Dictionary with extracted field at top level

    Example:
        >>> extract({"a": 1, "b": {"c": 2, "d": 3}}, "b", "c")
        {"a": 1, "c": 2, "b": {"d": 3}}
    """
    if parent_field not in data:
        return copy.deepcopy(data)

    parent = data[parent_field]
    if not isinstance(parent, dict) or child_field not in parent:
        return copy.deepcopy(data)

    result = copy.deepcopy(data)
    parent_copy = result[parent_field]

    # Extract child to top level
    result[child_field] = parent_copy.pop(child_field)

    # Update parent (or remove if empty)
    if parent_copy:
        result[parent_field] = parent_copy
    else:
        del result[parent_field]

    return result


def cast(data: dict[str, Any], field: str, target_type: str) -> dict[str, Any]:
    """
    Cast field to specified type.

    Args:
        data: Input dictionary
        field: Field to cast
        target_type: Target type ("int", "float", "string", "bool")

    Returns:
        Dictionary with casted field

    Raises:
        ValueError: If cast fails

    Example:
        >>> cast({"score": "0.75"}, "score", "float")
        {"score": 0.75}
    """
    if field not in data:
        return copy.deepcopy(data)

    result = copy.deepcopy(data)
    value = data[field]

    try:
        if target_type == "int":
            result[field] = int(value)
        elif target_type == "float":
            result[field] = float(value)
        elif target_type == "string" or target_type == "str":
            result[field] = str(value)
        elif target_type == "bool" or target_type == "boolean":
            # Handle common bool representations
            if isinstance(value, str):
                result[field] = value.lower() in ("true", "1", "yes", "on")
            else:
                result[field] = bool(value)
        else:
            raise ValueError(f"Unsupported cast type: {target_type}")
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to cast field '{field}' to {target_type}.\n"
            f"Value: {value} (type: {type(value).__name__})\n"
            f"Error: {e}"
        ) from e

    return result


def stringify(data: dict[str, Any], fields: list[str] | None = None) -> dict[str, Any]:
    """
    Convert dict/list field values to JSON strings for Excel compatibility.

    Args:
        data: Input dictionary
        fields: List of fields to stringify. If None, stringify all dict/list values.

    Returns:
        Dictionary with complex values converted to JSON strings

    Example:
        >>> stringify({"a": 1, "b": {"c": 2}}, ["b"])
        {"a": 1, "b": '{"c": 2}'}
        >>> stringify({"a": 1, "b": {"c": 2}})  # All complex values
        {"a": 1, "b": '{"c": 2}'}
    """
    result = copy.deepcopy(data)

    if fields is None:
        # Stringify all dict/list values
        for key, value in result.items():
            if isinstance(value, (dict, list)):
                result[key] = json.dumps(value, ensure_ascii=False, default=str)
    else:
        # Stringify only specified fields
        for field in fields:
            if field in result:
                value = result[field]
                if isinstance(value, (dict, list)):
                    result[field] = json.dumps(value, ensure_ascii=False, default=str)

    return result
