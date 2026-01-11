"""AggregateDataReshape plugin for collection-level data transformations."""

from __future__ import annotations

from typing import Any, ClassVar

from elspeth.core.operations import (
    exclude_fields,
    filter_fields,
    rename,
)


class AggregateDataReshape:
    """
    Collection-level data transformation plugin.

    Applies operations to collection data (dict of arrays).
    Reads from input_key, transforms, writes to output_key.

    Config options:
        input_key (str, required): Where to read collection from aggregates dict
        output_key (str, required): Where to write transformed collection
        operations (list, required): List of operation dictionaries

    Supported operations:
        - Same as RowDataReshape (operates on collection fields)
        - Operations work on field level (e.g., exclude_fields removes columns)

    Example:
        aggregation_plugins:
          - name: field_collector
            output_key: raw_data

          - name: aggregate_data_reshape
            input_key: raw_data
            output_key: clean_data
            operations:
              - exclude_fields: [prompt, response, metadata]
              - filter_fields: [id, score, delta]
    """

    # Config schema - validates YAML configuration
    config_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["input_key", "output_key", "operations"],
        "properties": {
            "input_key": {
                "type": "string",
                "description": "Key where collection is stored in aggregates dict"
            },
            "output_key": {
                "type": "string",
                "description": "Key where transformed collection will be stored"
            },
            "operations": {
                "type": "array",
                "description": "List of operations to apply",
                "items": {"type": "object"},
                "minItems": 1
            }
        }
    }

    # Input schema - accepts collection type
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "collection",
        "description": "Collection data (dict of arrays)"
    }

    # Output schema - produces collection type
    output_schema: ClassVar[dict[str, Any]] = {
        "type": "collection",
        "description": "Transformed collection data (dict of arrays)"
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize AggregateDataReshape with configuration."""
        self.config = config
        self.name = config.get("name", "aggregate_data_reshape")
        self.input_key = config["input_key"]
        self.output_key = config["output_key"]
        self.operations = config["operations"]

    def aggregate(
        self,
        results: list[dict[str, Any]],
        aggregates: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Apply operations to collection data.

        Args:
            results: Unused (reads from aggregates instead)
            aggregates: Dict containing collection at input_key

        Returns:
            Transformed collection (dict of arrays)

        Raises:
            KeyError: If input_key not in aggregates
            ValueError: If operation fails
        """
        # Validate input_key exists
        if self.input_key not in aggregates:
            available = list(aggregates.keys()) if aggregates else "<none>"
            raise KeyError(
                f"AggregateDataReshape: input_key '{self.input_key}' not found in aggregates.\n"
                f"Available keys: {available}"
            )

        collection = aggregates[self.input_key]

        if not collection:
            return {}

        # For collection operations, we work on the field level
        # (not item level like row operations)
        result: dict[str, Any] = collection

        for i, operation in enumerate(self.operations):
            if not isinstance(operation, dict):
                raise ValueError(
                    f"AggregateDataReshape: Operation {i} is not a dict.\n"
                    f"Got: {operation} (type: {type(operation).__name__})"
                )

            if len(operation) != 1:
                raise ValueError(
                    f"AggregateDataReshape: Operation {i} must have exactly one key.\n"
                    f"Got: {list(operation.keys())}"
                )

            op_name, op_args = next(iter(operation.items()))

            try:
                result = self._apply_operation(result, op_name, op_args)
            except Exception as e:
                raise ValueError(
                    f"AggregateDataReshape: Operation {i} ({op_name}) failed.\n"
                    f"Input collection fields: {list(collection.keys())}\n"
                    f"After previous operations: {list(result.keys())}\n"
                    f"Error: {e}"
                ) from e

        return result

    def _apply_operation(
        self,
        collection: dict[str, Any],
        op_name: str,
        op_args: Any
    ) -> dict[str, Any]:
        """
        Apply single operation to collection.

        For collections, operations work on fields (columns), not items (rows).
        """
        # Most operations work the same way on collections
        # (they operate on the dict keys, which are field names)

        result: dict[str, Any]

        if op_name == "filter_fields":
            if not isinstance(op_args, list):
                raise ValueError(f"filter_fields expects list, got {type(op_args)}")
            result = filter_fields(collection, op_args)

        elif op_name == "exclude_fields":
            if not isinstance(op_args, list):
                raise ValueError(f"exclude_fields expects list, got {type(op_args)}")
            result = exclude_fields(collection, op_args)

        elif op_name == "rename":
            if not isinstance(op_args, dict):
                raise ValueError(f"rename expects dict, got {type(op_args)}")
            result = rename(collection, op_args)

        # Note: flatten, extract, cast don't make sense for collection-level operations
        # (they operate on nested structures within items, not on collection structure)

        else:
            raise ValueError(
                f"Operation '{op_name}' not supported for collection-level reshape.\n"
                f"Supported: filter_fields, exclude_fields, rename\n"
                f"For row-level operations (flatten, extract, cast), use RowDataReshape plugin"
            )

        return result
