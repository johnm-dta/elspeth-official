"""Passthrough transform that logs row data for debugging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PassthroughLoggerPlugin:
    """Transform plugin that logs row data and passes through unchanged.

    Useful for debugging pipelines - see what data is flowing through
    at any point in the transform chain.

    Config options:
        log_level: "debug", "info", "warning" (default: "info")
        log_fields: List of field names to log (default: all)
        output_file: Optional file path to write JSONL output
        label: Optional label to identify this logger in output
    """

    name = "passthrough_logger"

    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.label = self.config.get("label", "passthrough")
        self.log_fields = self.config.get("log_fields")
        self.log_level = self.config.get("log_level", "info").lower()

        output_file = self.config.get("output_file")
        self.output_path = Path(output_file) if output_file else None
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._row_count = 0

    def transform(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Log row data and return unchanged."""
        self._row_count += 1

        # Filter fields if specified
        log_data = {k: v for k, v in row.items() if k in self.log_fields} if self.log_fields else row

        # Log to console
        msg = f"[{self.label}] Row {self._row_count}: {len(row)} fields"
        if self.log_level == "debug":
            logger.debug(msg)
            logger.debug(f"[{self.label}] Data: {log_data}")
        elif self.log_level == "warning":
            logger.warning(msg)
        else:
            logger.info(msg)

        # Write to file if configured
        if self.output_path:
            with self.output_path.open("a", encoding="utf-8") as f:
                record = {
                    "row_num": self._row_count,
                    "label": self.label,
                    "data": log_data,
                }
                f.write(json.dumps(record) + "\n")

        return row


# Register the plugin
from elspeth.core.sda.plugin_registry import register_transform_plugin

register_transform_plugin(
    "passthrough_logger",
    lambda options: PassthroughLoggerPlugin(options),
)
