"""Prompt template abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from jinja2 import Template, UndefinedError

from .exceptions import PromptRenderingError


@dataclass
class PromptTemplate:
    """Wrapper around a compiled Jinja template with helpful metadata."""

    name: str
    raw: str
    template: Template
    defaults: Mapping[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    required_fields: tuple[str, ...] = field(default_factory=tuple)

    def render(self, context: Mapping[str, Any] | None = None, extra: Mapping[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {}
        if self.defaults:
            payload.update(self.defaults)
        if context:
            payload.update(context)
        if extra:
            payload.update(extra)
        try:
            return self.template.render(**payload)
        except UndefinedError as exc:  # pragma: no cover - exercised via unit tests
            raise PromptRenderingError(str(exc), name=self.name) from exc

    def clone(
        self,
        *,
        name: str | None = None,
        raw: str | None = None,
        defaults: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        template: Template | None = None,
        required_fields: tuple[str, ...] | None = None,
    ) -> PromptTemplate:
        """Return a new instance with overrides applied."""

        return PromptTemplate(
            name=name or self.name,
            raw=raw or self.raw,
            template=template or self.template,
            defaults=defaults or self.defaults,
            metadata=dict(metadata or self.metadata),
            required_fields=required_fields or self.required_fields,
        )
