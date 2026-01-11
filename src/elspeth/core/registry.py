"""Simple registry for resolving plugin implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from elspeth.core.validation import ConfigurationError, validate_schema

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from elspeth.core.interfaces import DataSource, LLMClientProtocol, ResultSink

ON_ERROR_ENUM = {"type": "string", "enum": ["abort", "skip"]}

ARTIFACT_DESCRIPTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "type": {"type": "string"},
        "schema_id": {"type": "string"},
        "persist": {"type": "boolean"},
        "alias": {"type": "string"},
        "security_level": {"type": "string"},
    },
    "required": ["name", "type"],
    "additionalProperties": False,
}

ARTIFACTS_SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "produces": {
            "type": "array",
            "items": ARTIFACT_DESCRIPTOR_SCHEMA,
        },
        "consumes": {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "properties": {
                            "token": {"type": "string"},
                            "mode": {"type": "string", "enum": ["single", "all"]},
                        },
                        "required": ["token"],
                        "additionalProperties": False,
                    },
                ]
            },
        },
    },
    "additionalProperties": False,
}


@dataclass
class PluginFactory:
    create: Callable[[dict[str, Any]], Any]
    schema: Mapping[str, Any] | None = None

    def validate(self, options: dict[str, Any], context: str) -> None:
        if self.schema is None:
            return
        errors = list(validate_schema(options or {}, self.schema, context=context))
        if errors:
            message = "\n".join(msg.format() for msg in errors)
            raise ConfigurationError(message)


class PluginRegistry:
    def register_datasource(
        self, name: str, cls: type, schema: dict[str, Any] | None = None
    ) -> None:
        """Register a datasource plugin.

        Args:
            name: Plugin identifier used in config (e.g., 'local_csv')
            cls: The datasource class to instantiate
            schema: JSON Schema for validating plugin options
        """
        self._datasources[name] = PluginFactory(
            create=lambda options, c=cls: c(**options),  # type: ignore[misc]
            schema=schema,
        )

    def register_llm(
        self, name: str, cls: type, schema: dict[str, Any] | None = None
    ) -> None:
        """Register an LLM client plugin.

        Args:
            name: Plugin identifier used in config (e.g., 'azure_openai')
            cls: The LLM client class to instantiate
            schema: JSON Schema for validating plugin options
        """
        self._llms[name] = PluginFactory(
            create=lambda options, c=cls: c(**options),  # type: ignore[misc]
            schema=schema,
        )

    def register_sink(
        self, name: str, cls: type, schema: dict[str, Any] | None = None
    ) -> None:
        """Register an output sink plugin.

        Args:
            name: Plugin identifier used in config (e.g., 'csv')
            cls: The sink class to instantiate
            schema: JSON Schema for validating plugin options
        """
        self._sinks[name] = PluginFactory(
            create=lambda options, c=cls: c(**options),  # type: ignore[misc]
            schema=schema,
        )

    def reset(self) -> None:
        """Clear all plugin registrations. For testing only."""
        self._datasources.clear()
        self._llms.clear()
        self._sinks.clear()

    def __init__(self):
        self._datasources: dict[str, PluginFactory] = {}
        self._llms: dict[str, PluginFactory] = {}
        self._sinks: dict[str, PluginFactory] = {}

    def create_datasource(self, name: str, options: dict[str, Any]) -> DataSource:
        try:
            factory = self._datasources[name]
        except KeyError as exc:
            raise ValueError(f"Unknown datasource plugin '{name}'") from exc
        factory.validate(options or {}, context=f"datasource:{name}")
        return cast("DataSource", factory.create(options))

    def validate_datasource(self, name: str, options: dict[str, Any] | None) -> None:
        try:
            factory = self._datasources[name]
        except KeyError as exc:
            raise ValueError(f"Unknown datasource plugin '{name}'") from exc
        factory.validate(options or {}, context=f"datasource:{name}")

    def create_llm(self, name: str, options: dict[str, Any]) -> LLMClientProtocol:
        try:
            factory = self._llms[name]
        except KeyError as exc:
            raise ValueError(f"Unknown llm plugin '{name}'") from exc
        factory.validate(options or {}, context=f"llm:{name}")
        return cast("LLMClientProtocol", factory.create(options))

    def validate_llm(self, name: str, options: dict[str, Any] | None) -> None:
        try:
            factory = self._llms[name]
        except KeyError as exc:
            raise ValueError(f"Unknown llm plugin '{name}'") from exc
        factory.validate(options or {}, context=f"llm:{name}")

    def create_sink(self, name: str, options: dict[str, Any]) -> ResultSink:
        try:
            factory = self._sinks[name]
        except KeyError as exc:
            raise ValueError(f"Unknown sink plugin '{name}'") from exc
        factory.validate(options or {}, context=f"sink:{name}")
        return cast("ResultSink", factory.create(options))

    def validate_sink(self, name: str, options: dict[str, Any] | None) -> None:
        try:
            factory = self._sinks[name]
        except KeyError as exc:
            raise ValueError(f"Unknown sink plugin '{name}'") from exc
        factory.validate(options or {}, context=f"sink:{name}")


registry = PluginRegistry()

# Import datasources after registry creation to allow self-registration
# Both CSVDataSource and BlobDataSource register themselves when imported
from elspeth.plugins import datasources as _datasources  # noqa: F401

# Import LLM plugins after registry creation to allow self-registration
# All LLM clients register themselves when imported
from elspeth.plugins import llms as _llms  # noqa: F401

# Import output plugins after registry creation to allow self-registration
# All output sinks register themselves when imported
from elspeth.plugins import outputs as _outputs  # noqa: F401
