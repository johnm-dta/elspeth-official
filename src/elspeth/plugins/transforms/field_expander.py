"""FieldExpander meta-plugin for collection→row transposition."""

from __future__ import annotations

from typing import Any, ClassVar

from elspeth.core.sda.plugin_registry import register_aggregation_transform


class FieldExpander:
    """
    Meta-plugin: Converts columnar data back to row objects.

    Performs inverse operation of FieldCollector, enabling
    batch→row transitions in pipelines.

    Config options:
        input_key (str, required): Where to read collection from aggregates dict

    Example:
        aggregation_plugins:
          - name: field_expander
            input_key: "collected_metrics"
    """

    # Config schema - validates YAML configuration
    config_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["input_key"],
        "properties": {
            "input_key": {
                "type": "string",
                "description": "Key where collection is stored in aggregates dict"
            }
        }
    }

    # Input schema - accepts collection type
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "collection",
        "item_schema": {"type": "object"}
    }

    # Output schema - produces array of objects
    output_schema: ClassVar[dict[str, Any]] = {
        "type": "array",
        "items": {"type": "object"}
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize FieldExpander with configuration."""
        self.config = config
        self.name = "field_expander"

    def aggregate(
        self,
        results: list[dict[str, Any]],
        aggregates: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Transpose columnar format back to row objects.

        Args:
            results: Unused (reads from aggregates instead)
            aggregates: Dict containing collection at input_key

        Returns:
            Array of row objects
        """
        input_key = self.config['input_key']

        # Validate input_key exists
        if input_key not in aggregates:
            available = list(aggregates.keys()) if aggregates else "<none>"
            raise KeyError(
                f"FieldExpander: input_key '{input_key}' not found in aggregates.\n"
                f"Available keys: {available}"
            )

        collection = aggregates[input_key]

        if not collection:
            return []

        # Determine number of rows from first field
        fields = list(collection.keys())
        if not fields:
            return []

        num_rows = len(collection[fields[0]])

        # Validate all fields have same length
        for field in fields:
            field_length = len(collection[field])
            if field_length != num_rows:
                raise ValueError(
                    f"FieldExpander: Inconsistent array lengths in collection.\n"
                    f"Field '{fields[0]}' has {num_rows} items, "
                    f"but field '{field}' has {field_length} items"
                )

        # Transpose to row objects
        rows: list[dict[str, Any]] = []
        for i in range(num_rows):
            row = {field: collection[field][i] for field in fields}
            rows.append(row)

        return rows


# Register plugin
register_aggregation_transform(
    "field_expander",
    lambda options: FieldExpander(options),
    schema=FieldExpander.config_schema,
)
