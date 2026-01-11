"""Compile-time pipeline validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from elspeth.core.validation import ValidationMessage


@dataclass
class ValidationError(Exception):
    """Pipeline validation error."""
    messages: list[ValidationMessage]

    def __str__(self) -> str:
        return "\n".join(msg.format() for msg in self.messages)


class PipelineValidator:
    """Validates plugin pipelines at configuration time."""

    def validate_row_plugin_chain(self, plugins: list[Any]) -> list[ValidationMessage]:
        """
        Validate chain of row plugins.

        Checks that each plugin's output_schema is compatible with
        the next plugin's input_schema.
        """
        errors: list[ValidationMessage] = []

        for i in range(len(plugins) - 1):
            source = plugins[i]
            target = plugins[i + 1]

            source_output = getattr(source, 'output_schema', None)
            target_input = getattr(target, 'input_schema', None)

            if not source_output or not target_input:
                continue  # Can't validate without schemas

            # Check type compatibility
            source_type = source_output.get('type')
            target_type = target_input.get('type')

            if source_type and target_type and source_type != target_type:
                source_name = source.__class__.__name__
                target_name = target.__class__.__name__
                errors.append(ValidationMessage(
                    message=(
                        f"Type mismatch in row plugin chain: "
                        f"{source_name} outputs '{source_type}' but "
                        f"{target_name} expects '{target_type}'"
                    ),
                    context=f"row_plugins[{i}] -> row_plugins[{i+1}]"
                ))

        return errors

    def validate_execution_mode_transition(
        self,
        source: Any,
        target: Any
    ) -> list[ValidationMessage]:
        """
        Validate transition between execution modes.

        Checks if source plugin output type matches target plugin input type.
        """
        errors: list[ValidationMessage] = []

        source_output = getattr(source, 'output_schema', None)
        target_input = getattr(target, 'input_schema', None)

        if not source_output or not target_input:
            return errors

        source_type = source_output.get('type')
        target_type = target_input.get('type')

        if source_type != target_type:
            source_name = source.__class__.__name__
            target_name = target.__class__.__name__

            suggestion = ""
            if source_type == "object" and target_type == "collection":
                suggestion = " Insert field_collector to convert object → collection."
            elif source_type == "collection" and target_type == "array":
                suggestion = " Insert field_expander to convert collection → array."

            errors.append(ValidationMessage(
                message=(
                    f"Execution mode mismatch: "
                    f"{source_name} outputs '{source_type}' but "
                    f"{target_name} expects '{target_type}'.{suggestion}"
                ),
                context="pipeline"
            ))

        return errors

    def validate_dependency_graph(
        self,
        plugins: list[Any]
    ) -> list[ValidationMessage]:
        """
        Validate input_key/output_key dependency graph.

        Checks that all input_key references point to existing output_keys.
        """
        errors: list[ValidationMessage] = []

        # Build map of output_key → plugin
        outputs: dict[str, Any] = {}
        for plugin in plugins:
            config = getattr(plugin, 'config', {})
            output_key = config.get('output_key')
            if output_key:
                outputs[output_key] = plugin

        # Check all input_key references
        for plugin in plugins:
            config = getattr(plugin, 'config', {})
            input_key = config.get('input_key')

            if input_key and input_key not in outputs:
                plugin_name = plugin.__class__.__name__
                available = list(outputs.keys()) if outputs else "<none>"
                errors.append(ValidationMessage(
                    message=(
                        f"Plugin '{plugin_name}' requires input_key '{input_key}' "
                        f"but no plugin outputs to that key. "
                        f"Available keys: {available}"
                    ),
                    context="aggregation_plugins"
                ))

        return errors

    def validate_field_requirements(
        self,
        source: Any,
        target: Any
    ) -> list[ValidationMessage]:
        """
        Validate that source provides all required fields for target.

        Checks if source plugin's output includes all required fields
        from target plugin's input schema.
        """
        errors: list[ValidationMessage] = []

        source_output = getattr(source, 'output_schema', None)
        target_input = getattr(target, 'input_schema', None)

        if not source_output or not target_input:
            return errors

        # Only validate for object and collection types
        source_type = source_output.get('type')
        target_type = target_input.get('type')

        if source_type not in ('object', 'collection') or target_type not in ('object', 'collection'):
            return errors

        # Get source output fields
        source_props = source_output.get('properties', {})
        source_fields = set(source_props.keys())

        # For collection, check item_schema properties
        if source_type == 'collection':
            item_schema = source_output.get('item_schema', {})
            source_props = item_schema.get('properties', {})
            source_fields = set(source_props.keys())

        # Get target required fields
        target_required = target_input.get('required', [])
        if target_type == 'collection':
            item_schema = target_input.get('item_schema', {})
            target_required = item_schema.get('required', [])

        # Check for missing fields
        missing = set(target_required) - source_fields

        if missing:
            source_name = source.__class__.__name__
            target_name = target.__class__.__name__
            errors.append(ValidationMessage(
                message=(
                    f"Field requirements not met: "
                    f"{target_name} requires fields {sorted(missing)} "
                    f"but {source_name} doesn't provide them"
                ),
                context="pipeline"
            ))

        return errors
