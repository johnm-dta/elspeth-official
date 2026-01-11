# src/elspeth/plugins/transforms/row_data_reshape.py
"""RowDataReshape plugin for row-level data transformations."""

from __future__ import annotations

from typing import Any, ClassVar

from elspeth.core.operations import (
    cast,
    exclude_fields,
    extract,
    filter_fields,
    flatten,
    rename,
    stringify,
)


class RowDataReshape:
    """
    Row-level data transformation plugin.

    Applies sequence of operations to each row during processing.
    Operations are applied in order from shared operations library.

    Config options:
        operations (list, required): List of operation dictionaries

    Supported operations:
        - flatten: field_name
        - rename: {old: new, ...}
        - filter_fields: [field1, field2, ...]
        - exclude_fields: [field1, field2, ...]
        - extract: {parent: field, child: field}
        - cast: {field: field, type: target_type}
        - stringify: [field1, field2, ...] or null (all complex values)

    Example:
        row_plugins:
          - name: row_data_reshape
            operations:
              - flatten: metadata
              - rename:
                  metadata.confidence: confidence
              - exclude_fields: [prompt, response]
              - cast:
                  field: score
                  type: float
              - stringify: null  # Convert all dict/list values to JSON strings
    """

    # Config schema - validates YAML configuration
    config_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["operations"],
        "properties": {
            "operations": {
                "type": "array",
                "description": "List of operations to apply in sequence",
                "items": {"type": "object"},
                "minItems": 1
            }
        }
    }

    # Input schema - accepts objects (row-mode)
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Row data (object)"
    }

    # Output schema - produces objects (row-mode)
    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Transformed row data (object)"
    }

    # Indicates this plugin rewrites the entire row (not sparse metrics)
    rewrites_row: ClassVar[bool] = True

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize RowDataReshape with configuration."""
        self.config = config
        self.name = config.get("name", "row_data_reshape")
        self.operations = config["operations"]

    def transform(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        Apply operations to row data.

        Args:
            row: Input row (dict)
            context: Shared context for inter-plugin communication (unused)

        Returns:
            Transformed row (dict)

        Raises:
            ValueError: If operation is malformed or fails
        """
        result = row

        for i, operation in enumerate(self.operations):
            if not isinstance(operation, dict):
                raise ValueError(
                    f"RowDataReshape: Operation {i} is not a dict.\n"
                    f"Got: {operation} (type: {type(operation).__name__})\n"
                    f"Expected: dict with single key (operation name)"
                )

            if len(operation) != 1:
                raise ValueError(
                    f"RowDataReshape: Operation {i} must have exactly one key.\n"
                    f"Got: {list(operation.keys())}\n"
                    f"Expected: one of [flatten, rename, filter_fields, exclude_fields, extract, cast]"
                )

            op_name, op_args = next(iter(operation.items()))

            try:
                result = self._apply_operation(result, op_name, op_args)
            except Exception as e:
                raise ValueError(
                    f"RowDataReshape: Operation {i} ({op_name}) failed.\n"
                    f"Input: {row}\n"
                    f"After previous operations: {result}\n"
                    f"Error: {e}"
                ) from e

        return result

    def _apply_operation(
        self,
        data: dict[str, Any],
        op_name: str,
        op_args: Any
    ) -> dict[str, Any]:
        """Apply single operation to data."""
        if op_name == "flatten":
            if not isinstance(op_args, str):
                raise ValueError(f"flatten expects string field name, got {type(op_args)}")
            return flatten(data, op_args)

        elif op_name == "rename":
            if not isinstance(op_args, dict):
                raise ValueError(f"rename expects dict mapping, got {type(op_args)}")
            return rename(data, op_args)

        elif op_name == "filter_fields":
            if not isinstance(op_args, list):
                raise ValueError(f"filter_fields expects list of fields, got {type(op_args)}")
            return filter_fields(data, op_args)

        elif op_name == "exclude_fields":
            if not isinstance(op_args, list):
                raise ValueError(f"exclude_fields expects list of fields, got {type(op_args)}")
            return exclude_fields(data, op_args)

        elif op_name == "extract":
            if not isinstance(op_args, dict):
                raise ValueError(f"extract expects dict with 'parent' and 'child', got {type(op_args)}")
            parent = op_args.get("parent")
            child = op_args.get("child")
            if parent is None or child is None:
                raise ValueError("extract requires both 'parent' and 'child' fields")
            return extract(data, parent, child)

        elif op_name == "cast":
            if not isinstance(op_args, dict):
                raise ValueError(f"cast expects dict with 'field' and 'type', got {type(op_args)}")
            field = op_args.get("field")
            target_type = op_args.get("type")
            if field is None or target_type is None:
                raise ValueError("cast requires both 'field' and 'type' fields")
            return cast(data, field, target_type)

        elif op_name == "stringify":
            # op_args can be: null (all), list of field names, or single field name
            if op_args is None:
                return stringify(data, None)
            elif isinstance(op_args, list):
                return stringify(data, op_args)
            elif isinstance(op_args, str):
                return stringify(data, [op_args])
            else:
                raise ValueError(f"stringify expects null, list of fields, or field name, got {type(op_args)}")

        else:
            raise ValueError(
                f"Unknown operation: {op_name}\n"
                f"Supported: flatten, rename, filter_fields, exclude_fields, extract, cast, stringify"
            )


# Self-registration
from elspeth.core.sda.plugin_registry import register_transform_plugin

register_transform_plugin(
    "row_data_reshape",
    lambda config: RowDataReshape(config),
)
