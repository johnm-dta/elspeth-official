"""FieldCollector meta-plugin for row→collection transposition."""

from __future__ import annotations

from typing import Any, ClassVar

from elspeth.core.sda.plugin_registry import register_aggregation_transform


class FieldCollector:
    """
    Meta-plugin: Transposes row-based data to columnar format.

    Collects all row results and transposes to collection type
    (object with array values). Enables batch processing of row data.

    Config options:
        output_key (str, required): Where to write collection in aggregates dict
        exclude_fields (list[str], optional): Fields to exclude from collection

    Example:
        aggregation_plugins:
          - name: field_collector
            output_key: "collected_metrics"
            exclude_fields: [prompt, response, retry]
    """

    # Config schema - validates YAML configuration
    config_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["output_key"],
        "properties": {
            "output_key": {
                "type": "string",
                "description": "Key where collection will be stored in aggregates dict"
            },
            "exclude_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Fields to exclude from collection",
                "default": []
            }
        }
    }

    # Input schema - accepts array of objects (row results)
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "array",
        "items": {"type": "object"}
    }

    # Output schema - produces collection type
    # Note: item_schema is omitted because field types are dynamic
    output_schema: ClassVar[dict[str, Any]] = {
        "type": "collection"
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize FieldCollector with configuration."""
        self.config = config
        self.name = "field_collector"

    def aggregate(
        self,
        results: list[dict[str, Any]],
        aggregates: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Transpose row-based results to columnar format.

        Flattens nested metric fields from result["metrics"] into top-level
        arrays so analyzers can consume them directly. Recursively flattens
        nested dicts (e.g., scores: {quality: 0.8, accuracy: 0.9} becomes
        separate quality and accuracy arrays).

        Args:
            results: Array of row objects (all processed rows)
            aggregates: Growing dict of aggregation outputs (unused for collector)

        Returns:
            Columnar data: {field_name: [values...]}
        """
        if not results:
            return {}

        exclude = set(self.config.get('exclude_fields', []))

        # Collect union of all fields across ALL rows
        # Include top-level fields AND recursively flatten nested metrics
        fields: list[str] = []
        seen: set[str] = set()
        # Track nested paths: field_name -> list of (parent_key, nested_path) tuples
        nested_paths: dict[str, list[tuple[str | None, list[str]]]] = {}

        # Scan all rows to gather complete field list
        for row in results:
            # Add top-level fields (excluding "metrics" which we'll flatten)
            for key in row:
                if key not in exclude and key != "metrics":
                    value = row[key]
                    # Flatten top-level dicts that are metric-like (e.g., scores, score_flags)
                    # e.g., scores: {quality: 0.8, accuracy: 0.9} → quality, accuracy
                    # But preserve structural dicts like row, response
                    if isinstance(value, dict) and self._has_simple_values(value) and self._is_metric_dict(key):
                        self._scan_nested_dict(value, [], key, exclude, seen, fields, nested_paths, in_metrics=False)
                    elif key not in seen:
                        fields.append(key)
                        seen.add(key)
                        nested_paths[key] = [(None, [])]  # Direct top-level field

            # Add nested metric fields (recursively flatten dicts in metrics)
            if "metrics" in row and isinstance(row["metrics"], dict):
                for metric_key, metric_value in row["metrics"].items():
                    if metric_key in exclude:
                        continue
                    if isinstance(metric_value, dict):
                        # Recursively flatten nested dicts in metrics
                        # For example: metrics.scores.quality -> add "quality" field
                        self._scan_nested_dict(metric_value, [], metric_key, exclude, seen, fields, nested_paths, in_metrics=True)
                    elif metric_key not in seen:
                        # Simple metric value (not a dict)
                        fields.append(metric_key)
                        seen.add(metric_key)
                        nested_paths[metric_key] = [("metrics", [metric_key])]

        # Transpose to columnar format
        collection: dict[str, list[Any]] = {}

        for field in fields:
            values: list[Any] = []
            first_type: str | None = None

            for i, row in enumerate(results):
                # Try multiple paths to extract value
                value = self._extract_value(row, field, nested_paths.get(field, []))

                # Type consistency validation (skip None values and dicts)
                if value is not None and not isinstance(value, dict):
                    value_type = type(value).__name__
                    if first_type is None:
                        first_type = value_type
                    elif value_type != first_type:
                        raise TypeError(
                            f"FieldCollector: Type inconsistency in field '{field}'.\n"
                            f"First non-None value had type '{first_type}', but row {i} has type '{value_type}'.\n"
                            f"Field '{field}' values so far: {[v for v in values if v is not None]}\n"
                            f"Value at row {i}: {value}"
                        )

                values.append(value)

            collection[field] = values

        return collection

    def _has_simple_values(self, obj: dict[str, Any]) -> bool:
        """Check if dict contains only simple (non-dict) values at first level."""
        return all(not isinstance(value, dict) for value in obj.values())

    def _is_metric_dict(self, key: str) -> bool:
        """Check if a top-level dict key represents a metric dictionary.

        Metric dicts (e.g., scores, score_flags) should be flattened,
        while structural dicts (e.g., row, response) should be preserved.
        """
        # Common metric dict patterns
        metric_patterns = ["scores", "score_flags", "score_", "metric_", "_scores", "_metrics"]
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in metric_patterns)

    def _scan_nested_dict(
        self,
        obj: dict[str, Any],
        parent_path: list[str],
        parent_key: str,
        exclude: set[str],
        seen: set[str],
        fields: list[str],
        nested_paths: dict[str, list[tuple[str | None, list[str]]]],
        in_metrics: bool,
    ) -> None:
        """Recursively scan nested dict and add leaf fields.

        Args:
            in_metrics: True if scanning inside metrics dict, False if top-level
        """
        for key, value in obj.items():
            if key in exclude:
                continue

            current_path = [*parent_path, key]

            if isinstance(value, dict):
                # Recurse into nested dict
                self._scan_nested_dict(value, current_path, parent_key, exclude, seen, fields, nested_paths, in_metrics)
            else:
                # Leaf value - add field
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
                    nested_paths[key] = []

                # Track path to this field
                path = [parent_key, *current_path[:-1]]  # Exclude the key itself
                location = "metrics" if in_metrics else None
                nested_paths[key].append((location, path))

    def _extract_value(
        self,
        row: dict[str, Any],
        field: str,
        paths: list[tuple[str | None, list[str]]],
    ) -> Any:
        """Extract value from row following known paths.

        Tries nested/flattened paths first to give precedence to extracted
        fields over structural fields in case of name collisions.
        """
        # Try each known path first (nested/flattened paths)
        for location, path in paths:
            if location == "metrics":
                # Path from metrics dict
                current = row.get("metrics")
                if not isinstance(current, dict):
                    continue

                # Simple case: metrics.field (path = [field])
                if len(path) == 1 and path[0] == field:
                    return current.get(field)

                # Nested case: metrics.parent.child.field
                # Navigate nested path
                for key in path:
                    if not isinstance(current, dict) or key not in current:
                        current = None
                        break
                    current = current[key]

                # Extract final field value
                if isinstance(current, dict) and field in current:
                    return current[field]
            elif location is None:
                # Direct top-level field (path = [])
                if not path:
                    continue  # Already checked above

                # Path from top-level dict
                current = row.get(path[0])
                if not isinstance(current, dict):
                    continue

                # Navigate nested path (skip first element as we already got it)
                for key in path[1:]:
                    if not isinstance(current, dict) or key not in current:
                        current = None
                        break
                    current = current[key]

                # Extract final field value
                if isinstance(current, dict) and field in current:
                    return current[field]

        # Fall back to direct top-level access
        if field in row:
            return row[field]

        # Field missing in this row
        return None


# Register plugin
register_aggregation_transform(
    "field_collector",
    lambda options: FieldCollector(options),
    schema=FieldCollector.config_schema,
)
