"""Helpers for sink artifact type declarations."""

from __future__ import annotations

from typing import Any

FILE_PREFIX = "file/"
DATA_PREFIX = "data/"
VALID_PREFIXES = (FILE_PREFIX, DATA_PREFIX)


def is_file_type(type_name: str) -> bool:
    return type_name.startswith(FILE_PREFIX)


def is_data_type(type_name: str) -> bool:
    return type_name.startswith(DATA_PREFIX)


def validate_artifact_type(type_name: str) -> None:
    prefix = next((prefix for prefix in VALID_PREFIXES if type_name.startswith(prefix)), None)
    if prefix is None:
        raise ValueError(f"Unsupported artifact type '{type_name}'. Expected prefix one of {VALID_PREFIXES}.")
    suffix = type_name[len(prefix) :]
    if not suffix:
        raise ValueError(f"Artifact type '{type_name}' must include subtype, e.g. 'file/csv'.")


def normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata) if metadata else {}


__all__ = [
    "DATA_PREFIX",
    "FILE_PREFIX",
    "VALID_PREFIXES",
    "is_data_type",
    "is_file_type",
    "normalize_metadata",
    "validate_artifact_type",
]
