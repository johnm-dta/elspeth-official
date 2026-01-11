"""Secrets loading and variable substitution for configuration files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


class SecretsError(Exception):
    """Raised when secrets loading or substitution fails."""

    pass


def load_secrets(path: Path) -> dict[str, str]:
    """Load secrets from a YAML file.

    Args:
        path: Path to the secrets YAML file

    Returns:
        Dictionary of secret name -> value

    Raises:
        SecretsError: If file not found or invalid YAML
    """
    if not path.exists():
        raise SecretsError(f"Secrets file not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise SecretsError(f"Secrets file must contain a YAML mapping, got {type(data).__name__}")
        # Convert all values to strings
        return {str(k): str(v) for k, v in data.items()}
    except yaml.YAMLError as e:
        raise SecretsError(f"Invalid YAML in secrets file: {e}") from e


# Pattern matches ${VAR_NAME} - VAR_NAME can contain letters, digits, underscores
_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def substitute_variables(
    config: dict[str, Any],
    secrets: dict[str, str],
    *,
    _path: str = "",
) -> dict[str, Any]:
    """Substitute ${VAR} placeholders in config with secrets values.

    Args:
        config: Configuration dictionary (may be nested)
        secrets: Dictionary of variable name -> value
        _path: Internal - current path for error messages

    Returns:
        New config dict with all ${VAR} placeholders replaced

    Raises:
        SecretsError: If a referenced variable is not in secrets
    """
    result: dict[str, Any] = {}

    for key, value in config.items():
        current_path = f"{_path}.{key}" if _path else key
        result[key] = _substitute_value(value, secrets, current_path)

    return result


def _substitute_value(value: Any, secrets: dict[str, str], path: str) -> Any:
    """Substitute variables in a single value (recursive)."""
    if isinstance(value, dict):
        return {k: _substitute_value(v, secrets, f"{path}.{k}") for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_value(item, secrets, f"{path}[{i}]") for i, item in enumerate(value)]
    elif isinstance(value, str):
        return _substitute_string(value, secrets, path)
    else:
        return value


def _substitute_string(value: str, secrets: dict[str, str], path: str) -> str:
    """Substitute ${VAR} patterns in a string value."""
    # Handle escaped $$ -> $
    value = value.replace("$$", "\x00ESCAPED_DOLLAR\x00")

    # Find all ${VAR} references
    missing = []

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name not in secrets:
            missing.append(var_name)
            return match.group(0)  # Keep original for error message
        return secrets[var_name]

    result = _VAR_PATTERN.sub(replacer, value)

    if missing:
        raise SecretsError(
            f"Missing secret(s) at '{path}': {', '.join(missing)}\n"
            f"Add these to your secrets.yaml file"
        )

    # Restore escaped dollars
    return result.replace("\x00ESCAPED_DOLLAR\x00", "$")


def get_unresolved_variables(config: dict[str, Any]) -> list[str]:
    """Find all ${VAR} references in config that haven't been substituted.

    Useful for validation and error reporting.

    Args:
        config: Configuration dictionary

    Returns:
        List of variable names still present as ${VAR}
    """
    variables: list[str] = []
    _collect_variables(config, variables)
    return list(dict.fromkeys(variables))  # Dedupe preserving order


def _collect_variables(value: Any, variables: list[str]) -> None:
    """Recursively collect ${VAR} references."""
    if isinstance(value, dict):
        for v in value.values():
            _collect_variables(v, variables)
    elif isinstance(value, list):
        for item in value:
            _collect_variables(item, variables)
    elif isinstance(value, str):
        for match in _VAR_PATTERN.finditer(value):
            variables.append(match.group(1))
