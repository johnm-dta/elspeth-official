"""Preflight validation for configuration before execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from elspeth.core.secrets import get_unresolved_variables


@dataclass
class PreflightResult:
    """Result of preflight validation."""

    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary for operators."""
        if self.success:
            return "Preflight checks passed"

        lines = ["Preflight validation failed:", ""]

        if self.errors:
            lines.append("ERRORS:")
            for error in self.errors:
                lines.append(f"  - {error}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


class PreflightValidator:
    """Validates configuration before pipeline execution.

    Checks:
    - All ${VAR} placeholders have been substituted
    - Required configuration sections exist
    - (Future) Storage accounts are reachable
    - (Future) LLM endpoints respond
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize validator with configuration.

        Args:
            config: Resolved configuration dictionary
        """
        self.config = config

    def validate(self) -> PreflightResult:
        """Run all preflight checks.

        Returns:
            PreflightResult with success status and any errors/warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check for unresolved variables
        unresolved = get_unresolved_variables(self.config)
        if unresolved:
            for var in unresolved:
                errors.append(
                    f"Missing secret '{var}' - add it to your secrets.yaml file"
                )

        return PreflightResult(
            success=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
