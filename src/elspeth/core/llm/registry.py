"""Registry for LLM middleware plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elspeth.core.validation import ConfigurationError, validate_schema

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .middleware import LLMMiddleware


class _Factory:
    def __init__(self, factory: Callable[[dict[str, Any]], LLMMiddleware], schema: Mapping[str, Any] | None = None):
        self.factory = factory
        self.schema = schema

    def validate(self, options: dict[str, Any], *, context: str) -> None:
        if self.schema is None:
            return
        errors = list(validate_schema(options or {}, self.schema, context=context))
        if errors:
            raise ConfigurationError("\n".join(msg.format() for msg in errors))

    def create(self, options: dict[str, Any], *, context: str) -> LLMMiddleware:
        self.validate(options, context=context)
        return self.factory(options)


_middlewares: dict[str, _Factory] = {}


def register_middleware(
    name: str,
    factory: Callable[[dict[str, Any]], LLMMiddleware],
    *,
    schema: Mapping[str, Any] | None = None,
) -> None:
    _middlewares[name] = _Factory(factory, schema=schema)


def create_middleware(definition: dict[str, Any]) -> LLMMiddleware:
    if not definition:
        raise ValueError("Middleware definition cannot be empty")
    name = definition.get("name") or definition.get("plugin")
    if not name:
        raise ValueError("Middleware definition missing 'name' or 'plugin'")
    if name not in _middlewares:
        raise ValueError(f"Unknown LLM middleware '{name}'")
    options = definition.get("options", {})
    return _middlewares[name].create(options, context=f"llm_middleware:{name}")


def create_middlewares(definitions: list[dict[str, Any]] | None) -> list[LLMMiddleware]:
    if not definitions:
        return []
    return [create_middleware(defn) for defn in definitions]


def validate_middleware_definition(definition: dict[str, Any]) -> None:
    if not definition:
        raise ConfigurationError("Middleware definition cannot be empty")
    name = definition.get("name") or definition.get("plugin")
    if not name:
        raise ConfigurationError("Middleware definition missing 'name' or 'plugin'")
    if name not in _middlewares:
        options = ", ".join(sorted(_middlewares)) or "<none>"
        raise ConfigurationError(f"Unknown LLM middleware '{name}'. Available: {options}")
    options = definition.get("options", {})
    _middlewares[name].validate(options, context=f"llm_middleware:{name}")


__all__ = ["create_middleware", "create_middlewares", "register_middleware", "validate_middleware_definition"]
