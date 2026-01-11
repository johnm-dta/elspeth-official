"""Single row processing for SDA execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from elspeth.core.sda.plugins import TransformPlugin


class RowProcessor:
    """Processes single rows through transform plugins.

    This is a generic plugin executor - it has no knowledge of LLMs,
    prompts, or criteria. All domain-specific logic lives in plugins.
    """

    def __init__(
        self,
        transform_plugins: list[TransformPlugin],
        security_level: str | None = None,
    ) -> None:
        """Initialize row processor.

        Args:
            transform_plugins: Transform plugins to apply in sequence
            security_level: Security level for records
        """
        self.transform_plugins = transform_plugins
        self.security_level = security_level

    def process_row(
        self,
        row: pd.Series,
        row_data: dict[str, Any],
        row_id: str | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Process single row through transform plugins.

        Args:
            row: Pandas Series representing row (for compatibility)
            row_data: Row data as dictionary
            row_id: Unique row identifier

        Returns:
            Tuple of (record, failure). One will be None.
        """
        try:
            context: dict[str, Any] = {}

            # Apply transform plugins in sequence
            for plugin in self.transform_plugins:
                # Optional: validate input schema
                if hasattr(plugin, "input_schema") and plugin.input_schema:
                    from elspeth.core.validation import ConfigurationError, validate_schema

                    errors = list(validate_schema(row_data, plugin.input_schema))
                    if errors:
                        plugin_name = getattr(plugin, "name", plugin.__class__.__name__)
                        error_messages = "\n".join(f"  - {e.format()}" for e in errors)
                        raise ConfigurationError(
                            f"Input validation failed for {plugin_name}:\n{error_messages}"
                        )

                row_data = plugin.transform(row_data, context)

                # Optional: validate output schema
                if hasattr(plugin, "output_schema") and plugin.output_schema:
                    from elspeth.core.validation import ConfigurationError, validate_schema

                    errors = list(validate_schema(row_data, plugin.output_schema))
                    if errors:
                        plugin_name = getattr(plugin, "name", plugin.__class__.__name__)
                        error_messages = "\n".join(f"  - {e.format()}" for e in errors)
                        raise ConfigurationError(
                            f"Output validation failed for {plugin_name}:\n{error_messages}"
                        )

            record: dict[str, Any] = {
                "row": row_data,
                "context": context,
            }

            if self.security_level:
                record["security_level"] = self.security_level

            return record, None

        except Exception as exc:
            import time

            failure = {
                "row": row_data,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "timestamp": time.time(),
            }
            return None, failure
