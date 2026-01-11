"""Custom exceptions for prompt rendering."""

from __future__ import annotations


class PromptError(RuntimeError):
    """Base class for prompt-related failures."""


class PromptValidationError(PromptError):
    """Raised when a prompt template fails validation."""

    def __init__(self, message: str, *, missing: set[str] | None = None, name: str | None = None):
        super().__init__(message)
        self.missing = missing or set()
        self.name = name


class PromptRenderingError(PromptError):
    """Raised when rendering fails at runtime (e.g., missing variables)."""

    def __init__(self, message: str, *, name: str | None = None):
        super().__init__(message)
        self.name = name
