"""Prompt engine built on top of Jinja2."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, StrictUndefined
from jinja2 import meta as jinja_meta

from .exceptions import PromptValidationError
from .template import PromptTemplate

if TYPE_CHECKING:
    from collections.abc import Mapping

_FORMAT_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _auto_convert(text: str) -> str:
    """Convert simple ``{field}`` tokens to ``{{ field }}`` for compatibility."""

    if "{{" in text or "{%" in text:
        return text

    def repl(match: re.Match[str]) -> str:
        return "{{ " + match.group(1) + " }}"

    return _FORMAT_PATTERN.sub(repl, text)


def _create_environment() -> Environment:
    env = Environment(
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    env.filters.setdefault("default", _default_filter)
    env.filters.setdefault("upper", lambda value: value.upper() if isinstance(value, str) else value)
    env.filters.setdefault("lower", lambda value: value.lower() if isinstance(value, str) else value)
    env.filters.setdefault("title", lambda value: value.title() if isinstance(value, str) else value)
    return env


def _default_filter(value: Any, fallback: Any = "", boolean: bool = False) -> Any:
    if boolean:
        return value or fallback
    return fallback if value is None else value


@dataclass
class PromptEngine:
    environment: Environment = field(default_factory=_create_environment)

    def compile(
        self,
        source: str,
        *,
        name: str = "prompt",
        defaults: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PromptTemplate:
        normalized = _auto_convert(source or "")
        template = self.environment.from_string(normalized)
        ast = self.environment.parse(normalized)
        candidate_fields = jinja_meta.find_undeclared_variables(ast)
        required = tuple(sorted(self._filter_declared(candidate_fields)))
        if required and defaults:
            missing = {field for field in required if field not in defaults}
            if missing:
                # Validation will ultimately happen during render, but provide early signal when defaults missing.
                pass
        return PromptTemplate(
            name=name,
            raw=source or "",
            template=template,
            defaults=defaults or {},
            metadata=dict(metadata or {}),
            required_fields=required,
        )

    def validate(
        self,
        template: PromptTemplate,
        context: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        context = context or {}
        extra = extra or {}
        provided = set(template.defaults.keys()) | set(context.keys()) | set(extra.keys())
        missing = [field for field in template.required_fields if field not in provided]
        if missing:
            raise PromptValidationError(
                f"Missing fields for prompt '{template.name}': {', '.join(missing)}",
                missing=set(missing),
                name=template.name,
            )

    def render(
        self,
        template: PromptTemplate,
        context: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        return template.render(context=context, extra=extra)

    def _filter_declared(self, variables: set[str]) -> set[str]:
        skip = set(self.environment.globals.keys()) | {"loop", "cycler", "namespace"}
        return {var for var in variables if var not in skip}
