"""Diagnostics writer for error reporting and debugging."""

from __future__ import annotations

import re
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


class DiagnosticsWriter:
    """Writes diagnostic information for error investigation.

    Creates a diagnostics directory with:
    - error-summary.txt: Human-readable error description
    - config-resolved.yaml: Configuration with secrets redacted
    - stack-trace.txt: Full Python stack trace
    """

    # Patterns that indicate sensitive data
    _SENSITIVE_PATTERNS = [
        re.compile(r"(api[_-]?key)", re.IGNORECASE),
        re.compile(r"(secret)", re.IGNORECASE),
        re.compile(r"(password)", re.IGNORECASE),
        re.compile(r"(token)", re.IGNORECASE),
        re.compile(r"(connection[_-]?string)", re.IGNORECASE),
        re.compile(r"(account[_-]?key)", re.IGNORECASE),
        re.compile(r"(private[_-]?key)", re.IGNORECASE),
        re.compile(r"(credential)", re.IGNORECASE),
    ]

    def __init__(self, output_dir: Path) -> None:
        """Initialize diagnostics writer.

        Args:
            output_dir: Directory to write diagnostic files
        """
        self.output_dir = Path(output_dir)

    def _ensure_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_error(
        self,
        error_type: str,
        message: str,
        context: dict[str, Any],
    ) -> Path:
        """Write human-readable error summary.

        Args:
            error_type: Type/class of error
            message: Error message
            context: Additional context (file paths, etc.)

        Returns:
            Path to written file
        """
        self._ensure_dir()

        timestamp = datetime.now(UTC).isoformat()

        lines = [
            "=" * 60,
            "ELSPETH PIPELINE ERROR",
            "=" * 60,
            "",
            f"Timestamp: {timestamp}",
            f"Error Type: {error_type}",
            "",
            "Message:",
            f"  {message}",
            "",
        ]

        if context:
            lines.append("Context:")
            for key, value in context.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        lines.extend([
            "-" * 60,
            "Send this file to support for investigation.",
            "-" * 60,
        ])

        output_path = self.output_dir / "error-summary.txt"
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def write_config(self, config: dict[str, Any]) -> Path:
        """Write configuration with secrets redacted.

        Args:
            config: Resolved configuration dictionary

        Returns:
            Path to written file
        """
        self._ensure_dir()

        redacted = self._redact_secrets(config)

        output_path = self.output_dir / "config-resolved.yaml"
        output_path.write_text(
            yaml.safe_dump(redacted, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        return output_path

    def write_stack_trace(self) -> Path:
        """Write current exception stack trace.

        Call this from within an except block.

        Returns:
            Path to written file
        """
        self._ensure_dir()

        exc_info = sys.exc_info()
        if exc_info[0] is None:
            trace_text = "No exception currently being handled."
        else:
            trace_text = "".join(traceback.format_exception(*exc_info))

        output_path = self.output_dir / "stack-trace.txt"
        output_path.write_text(trace_text, encoding="utf-8")
        return output_path

    def _redact_secrets(self, value: Any, key: str = "") -> Any:
        """Recursively redact sensitive values.

        Args:
            value: Value to potentially redact
            key: Current key name (for pattern matching)

        Returns:
            Value with secrets replaced by [REDACTED]
        """
        if isinstance(value, dict):
            return {k: self._redact_secrets(v, k) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._redact_secrets(item, key) for item in value]
        elif isinstance(value, str):
            # Check if key name suggests this is sensitive
            if any(pattern.search(key) for pattern in self._SENSITIVE_PATTERNS):
                return "[REDACTED]"
            # Also redact if value looks like a secret (long base64, starts with sk-, etc.)
            if self._looks_like_secret(value):
                return "[REDACTED]"
            return value
        else:
            return value

    def _looks_like_secret(self, value: str) -> bool:
        """Heuristically detect if a string looks like a secret."""
        # API keys often start with these prefixes
        secret_prefixes = ("sk-", "pk-", "api-", "key-", "token-")
        if any(value.startswith(prefix) for prefix in secret_prefixes):
            return True

        # Connection strings contain AccountKey=
        return "AccountKey=" in value
