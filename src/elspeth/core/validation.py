"""Validation utilities and error reporting helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(RuntimeError):
    """Raised when configuration validation fails."""


@dataclass
class ValidationMessage:
    message: str
    context: str | None = None

    def format(self) -> str:
        if self.context:
            return f"{self.context}: {self.message}"
        return self.message


@dataclass
class ValidationReport:
    errors: list[ValidationMessage] = field(default_factory=list)
    warnings: list[ValidationMessage] = field(default_factory=list)

    def add_error(self, message: str, context: str | None = None) -> None:
        self.errors.append(ValidationMessage(message=message, context=context))

    def add_warning(self, message: str, context: str | None = None) -> None:
        self.warnings.append(ValidationMessage(message=message, context=context))

    def extend(self, other: ValidationReport) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def raise_if_errors(self) -> None:
        if self.errors:
            formatted = "\n".join(msg.format() for msg in self.errors)
            raise ConfigurationError(formatted)

    def has_errors(self) -> bool:
        return bool(self.errors)

    def has_warnings(self) -> bool:
        return bool(self.warnings)


def validate_schema(
    data: Mapping[str, object] | None,
    schema: Mapping[str, object],
    *,
    context: str | None = None,
) -> Iterable[ValidationMessage]:
    """Validate ``data`` against ``schema`` returning validation messages."""

    if data is None:
        yield ValidationMessage("value is missing", context=context)
        return

    errors: list[tuple[Sequence[object], str]] = []
    _validate_node(data, schema, (), errors)
    for path, message in errors:
        pointer = _format_error_path(path)
        details = message
        if pointer:
            details = f"{details} (path: {pointer})"
        yield ValidationMessage(details, context=context)


def _validate_node(value: Any, schema: Mapping[str, Any] | None, path: Sequence[object], errors: list[tuple[Sequence[object], str]]) -> None:
    if schema is None:
        return

    any_of = schema.get("anyOf")
    if any_of:
        for option in any_of:
            option_errors: list[tuple[Sequence[object], str]] = []
            _validate_node(value, option, path, option_errors)
            if not option_errors:
                break
        else:
            errors.append((path, "did not match any allowed schemas"))

    expected_type = schema.get("type")
    if expected_type and not _check_type(value, expected_type):
        errors.append((path, f"must be of type {expected_type}"))
        return

    # Collection type validation
    if expected_type == "collection" and isinstance(value, Mapping):
        item_schema = schema.get("item_schema")
        if item_schema:
            # Validate all array values in the collection
            for key, array_value in value.items():
                if not isinstance(array_value, list):
                    errors.append(((*path, key), "collection field must be an array"))
                    continue
                # Validate each element in the array against item_schema
                for index, item in enumerate(array_value):
                    _validate_node(item, item_schema, (*path, key, index), errors)

    if isinstance(value, Mapping) and expected_type != "collection":
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(((*path, key), "is a required property"))
        properties = schema.get("properties", {})
        for key, subschema in properties.items():
            if key in value:
                _validate_node(value[key], subschema, (*path, key), errors)

    if isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_node(item, item_schema, (*path, index), errors)

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        errors.append((path, f"must be one of {enum}"))

    minimum = schema.get("minimum")
    if minimum is not None and _is_number(value) and value < minimum:
        errors.append((path, f"must be >= {minimum}"))

    exclusive_min = schema.get("exclusiveMinimum")
    if exclusive_min is not None and _is_number(value) and value <= exclusive_min:
        errors.append((path, f"must be > {exclusive_min}"))


def _check_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return _is_number(value)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "collection":
        return isinstance(value, Mapping)  # Collection is an object with array values
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_error_path(path: Iterable[object]) -> str:
    parts = []
    for item in path:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            if parts:
                parts.append(".")
            parts.append(str(item))
    return "".join(parts)


def validate_settings(path: str | Path, profile: str = "default") -> ValidationReport:
    report = ValidationReport()
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        report.add_error("Settings file not found", context=str(config_path))
        return report
    except yaml.YAMLError as exc:  # pragma: no cover - invalid yaml
        report.add_error(f"Invalid YAML: {exc}", context=str(config_path))
        return report

    profile_data = raw.get(profile)
    if profile_data is None:
        report.add_error(f"Profile '{profile}' not found", context=str(config_path))
        return report

    for message in validate_schema(profile_data, _SETTINGS_SCHEMA, context=f"settings[{profile}]"):
        report.errors.append(message)
    if report.has_errors():
        return report

    from elspeth.core import registry as core_registry
    from elspeth.core.controls import registry as controls_registry
    from elspeth.core.llm import registry as llm_registry
    from elspeth.core.sda import plugin_registry as exp_registry

    registry = core_registry.registry

    _validate_plugin_reference(
        report,
        profile_data.get("datasource"),
        kind="datasource",
        validator=registry.validate_datasource,
    )
    # LLM is optional when using llm_query plugins in row_plugins
    llm_config = profile_data.get("llm")
    if llm_config is not None:
        _validate_plugin_reference(
            report,
            llm_config,
            kind="llm",
            validator=registry.validate_llm,
        )

    sinks = profile_data.get("sinks")
    if not isinstance(sinks, list) or not sinks:
        report.add_error("'sinks' must be a non-empty list", context=f"settings[{profile}]")
    else:
        for entry in sinks:
            _validate_plugin_reference(
                report,
                entry,
                kind="sink",
                validator=registry.validate_sink,
            )

    prompt_packs = profile_data.get("prompt_packs") or {}
    if not isinstance(prompt_packs, Mapping):
        report.add_error("'prompt_packs' must be a mapping", context=f"settings[{profile}]")
        prompt_packs = {}
    for name, pack in prompt_packs.items():
        _validate_prompt_pack(report, name, pack, registry, exp_registry, llm_registry)

    _validate_middleware_list(
        report,
        profile_data.get("llm_middlewares"),
        llm_registry.validate_middleware_definition,
        context=f"settings[{profile}].middleware",
    )

    prompt_pack_name = profile_data.get("prompt_pack")
    # Skip validation for remote URLs - they are resolved at runtime
    from elspeth.core.prompts.resolvers import is_remote_pack_url

    if prompt_pack_name and not is_remote_pack_url(prompt_pack_name) and prompt_pack_name not in prompt_packs:
        available = ", ".join(sorted(prompt_packs)) or "<none>"
        report.add_error(
            f"Unknown prompt pack '{prompt_pack_name}'. Available prompt packs: {available}",
            context=f"settings[{profile}].prompt_pack",
        )

    suite_defaults = profile_data.get("suite_defaults") or {}
    if not isinstance(suite_defaults, Mapping):
        report.add_error("'suite_defaults' must be a mapping", context=f"settings[{profile}]")
        suite_defaults = {}
    _validate_suite_defaults(report, suite_defaults, registry, exp_registry, llm_registry, controls_registry)

    for key in ("retry", "checkpoint", "concurrency"):
        value = profile_data.get(key)
        if value is not None and not isinstance(value, Mapping):
            report.add_error(f"'{key}' must be a mapping", context=f"settings[{profile}]")

    return report


def validate_suite(
    suite_root: str | Path,
    *,
    defaults: Mapping[str, Any] | None = None,
    row_estimate: int = 100,
) -> SuiteValidationReport:
    report = ValidationReport()
    suite_path = Path(suite_root)
    if not suite_path.exists():
        report.add_error("Suite root does not exist", context=str(suite_path))
        return SuiteValidationReport(report=report)

    from elspeth.core import registry as core_registry
    from elspeth.core.controls import registry as controls_registry
    from elspeth.core.llm import registry as llm_registry
    from elspeth.core.sda import plugin_registry as exp_registry

    registry = core_registry.registry

    experiments: list[dict[str, Any]] = []
    names: list[str] = []
    baseline_count = 0
    baseline_name: str | None = None

    for folder in sorted(p for p in suite_path.iterdir() if p.is_dir() and not p.name.startswith(".")):
        config_path = folder / "config.json"
        if not config_path.exists():
            continue
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.add_error(f"Invalid JSON: {exc}", context=str(config_path))
            continue

        for message in validate_schema(data, _EXPERIMENT_SCHEMA, context=f"experiment:{folder.name}"):
            report.errors.append(message)
        name = data.get("name") or folder.name
        enabled = bool(data.get("enabled", True))
        # Check both old location and new metadata location
        is_baseline = bool(data.get("is_baseline", False))
        if not is_baseline and "metadata" in data:
            is_baseline = bool(data["metadata"].get("is_baseline", False))
        if enabled:
            names.append(name)
            if is_baseline:
                baseline_count += 1
                baseline_name = name

        _validate_experiment_plugins(
            report, data.get("row_plugins"), exp_registry.validate_row_plugin_definition, f"experiment:{name}.row_plugin"
        )
        _validate_experiment_plugins(
            report,
            data.get("aggregator_plugins"),
            exp_registry.validate_aggregation_plugin_definition,
            f"experiment:{name}.aggregation_plugin",
        )
        _validate_experiment_plugins(
            report, data.get("baseline_plugins"), exp_registry.validate_baseline_plugin_definition, f"experiment:{name}.baseline_plugin"
        )
        _validate_experiment_plugins(
            report,
            data.get("early_stop_plugins"),
            exp_registry.validate_early_stop_plugin_definition,
            f"experiment:{name}.early_stop_plugin",
        )
        _validate_middleware_list(
            report, data.get("llm_middlewares"), llm_registry.validate_middleware_definition, context=f"experiment:{name}.middleware"
        )
        _validate_plugin_list(report, data.get("sinks"), registry.validate_sink, context=f"experiment:{name}.sink")
        try:
            controls_registry.validate_rate_limiter(data.get("rate_limiter"))
        except ConfigurationError as exc:
            report.add_error(str(exc), context=f"experiment:{name}.rate_limiter")
        try:
            controls_registry.validate_cost_tracker(data.get("cost_tracker"))
        except ConfigurationError as exc:
            report.add_error(str(exc), context=f"experiment:{name}.cost_tracker")

        if data.get("concurrency") is not None and not isinstance(data.get("concurrency"), Mapping):
            report.add_error("'concurrency' must be a mapping", context=f"experiment:{name}")

        _validate_prompt_files(report, folder, name, data)

        experiments.append(
            {
                "name": name,
                "enabled": enabled,
                "is_baseline": is_baseline,
                "criteria": data.get("criteria") or [],
                "temperature": data.get("temperature", 0.0),
                "max_tokens": data.get("max_tokens", 0),
            }
        )

    if not experiments:
        report.add_error("No experiments found", context=str(suite_path))

    duplicates = _find_duplicates(names)
    for dup in duplicates:
        report.add_error(f"Duplicate experiment name '{dup}'", context="suite")

    # Baseline is optional - only required for experimental orchestrator
    # StandardOrchestrator doesn't need one
    if baseline_count == 0:
        report.add_warning("No baseline experiment found (required for experimental orchestrator)", context="suite")

    warnings = []
    for exp in experiments:
        if exp["enabled"] and exp["temperature"] > 1.5:
            warning = f"High temperature ({exp['temperature']}) for experiment '{exp['name']}'"
            report.add_warning(warning, context="suite")
            warnings.append(warning)
        if exp["enabled"] and exp["max_tokens"] > 2000:
            warning = f"High max_tokens ({exp['max_tokens']}) for experiment '{exp['name']}'"
            report.add_warning(warning, context="suite")
            warnings.append(warning)

    enabled_experiments = [exp for exp in experiments if exp["enabled"]]
    criteria_counts = [len(exp["criteria"]) or 1 for exp in enabled_experiments]
    estimated_calls = sum(row_estimate * count for count in criteria_counts)
    estimated_time_minutes = estimated_calls / 60 if estimated_calls else 0

    preflight = {
        "experiment_count": len(enabled_experiments),
        "baseline": baseline_name,
        "estimated_api_calls": estimated_calls,
        "estimated_time_minutes": estimated_time_minutes,
        "warnings": warnings,
    }

    return SuiteValidationReport(report=report, preflight=preflight)


def _validate_plugin_reference(
    report: ValidationReport,
    entry: Any,
    *,
    kind: str,
    validator: Callable[[str, dict[str, Any] | None], None],
) -> None:
    if not isinstance(entry, Mapping):
        report.add_error(f"{kind} configuration must be a mapping", context=kind)
        return
    plugin = entry.get("plugin")
    if not plugin:
        report.add_error("Missing 'plugin'", context=kind)
        return
    if not isinstance(plugin, str):
        report.add_error("Plugin name must be a string", context=kind)
        return
    options = entry.get("options")
    try:
        validator(plugin, dict(options) if isinstance(options, Mapping) else {})
    except (ValueError, ConfigurationError) as exc:
        report.add_error(str(exc), context=f"{kind}:{plugin}")


def _validate_plugin_list(
    report: ValidationReport,
    entries: Any,
    validator: Callable[[str, dict[str, Any] | None], None],
    *,
    context: str,
) -> None:
    if entries is None:
        return
    if not isinstance(entries, list):
        report.add_error("Expected a list of plugin definitions", context=context)
        return
    for entry in entries:
        _validate_plugin_reference(report, entry, kind=context, validator=validator)


def _validate_prompt_pack(
    report: ValidationReport,
    name: str,
    pack: Any,
    registry,
    exp_registry,
    llm_registry,
) -> None:
    context = f"prompt_pack:{name}"
    if not isinstance(pack, Mapping):
        report.add_error("Prompt pack must be a mapping", context=context)
        return
    prompts = pack.get("prompts")
    if prompts is not None:
        if not isinstance(prompts, Mapping):
            report.add_error("'prompts' must be a mapping", context=context)
        else:
            if "system" not in prompts or "user" not in prompts:
                report.add_error("Prompt pack prompts must include 'system' and 'user'", context=context)

    _validate_experiment_plugins(report, pack.get("row_plugins"), exp_registry.validate_row_plugin_definition, f"{context}.row_plugin")
    _validate_experiment_plugins(
        report, pack.get("aggregator_plugins"), exp_registry.validate_aggregation_plugin_definition, f"{context}.aggregation_plugin"
    )
    _validate_experiment_plugins(
        report, pack.get("baseline_plugins"), exp_registry.validate_baseline_plugin_definition, f"{context}.baseline_plugin"
    )
    _validate_experiment_plugins(
        report, pack.get("early_stop_plugins"), exp_registry.validate_early_stop_plugin_definition, f"{context}.early_stop_plugin"
    )
    _validate_middleware_list(
        report, pack.get("llm_middlewares"), llm_registry.validate_middleware_definition, context=f"{context}.middleware"
    )
    _validate_plugin_list(report, pack.get("sinks"), registry.validate_sink, context=f"{context}.sink")


def _validate_suite_defaults(
    report: ValidationReport,
    defaults: Mapping[str, Any],
    registry,
    exp_registry,
    llm_registry,
    controls_registry,
) -> None:
    _validate_experiment_plugins(
        report, defaults.get("row_plugins"), exp_registry.validate_row_plugin_definition, "suite_defaults.row_plugin"
    )
    _validate_experiment_plugins(
        report, defaults.get("aggregator_plugins"), exp_registry.validate_aggregation_plugin_definition, "suite_defaults.aggregation_plugin"
    )
    _validate_experiment_plugins(
        report, defaults.get("baseline_plugins"), exp_registry.validate_baseline_plugin_definition, "suite_defaults.baseline_plugin"
    )
    _validate_experiment_plugins(
        report, defaults.get("early_stop_plugins"), exp_registry.validate_early_stop_plugin_definition, "suite_defaults.early_stop_plugin"
    )
    _validate_middleware_list(
        report, defaults.get("llm_middlewares"), llm_registry.validate_middleware_definition, context="suite_defaults.middleware"
    )
    _validate_plugin_list(report, defaults.get("sinks"), registry.validate_sink, context="suite_defaults.sink")
    try:
        controls_registry.validate_rate_limiter(defaults.get("rate_limiter"))
    except ConfigurationError as exc:
        report.add_error(str(exc), context="suite_defaults.rate_limiter")
    try:
        controls_registry.validate_cost_tracker(defaults.get("cost_tracker"))
    except ConfigurationError as exc:
        report.add_error(str(exc), context="suite_defaults.cost_tracker")


def _validate_experiment_plugins(
    report: ValidationReport,
    entries: Any,
    validator: Callable[[dict[str, Any]], None],
    context: str,
) -> None:
    if entries is None:
        return
    if not isinstance(entries, list):
        report.add_error("Expected a list of plugin definitions", context=context)
        return
    for definition in entries:
        if not isinstance(definition, Mapping):
            report.add_error("Plugin definition must be a mapping", context=context)
            continue
        try:
            validator(dict(definition))
        except ConfigurationError as exc:
            report.add_error(str(exc), context=context)
        except ValueError as exc:
            report.add_error(str(exc), context=context)


def _validate_middleware_list(
    report: ValidationReport,
    entries: Any,
    validator: Callable[[dict[str, Any]], None],
    *,
    context: str,
) -> None:
    if entries is None:
        return
    if not isinstance(entries, list):
        report.add_error("Expected a list of middleware definitions", context=context)
        return
    for definition in entries:
        if not isinstance(definition, Mapping):
            report.add_error("Middleware definition must be a mapping", context=context)
            continue
        try:
            validator(dict(definition))
        except ConfigurationError as exc:
            report.add_error(str(exc), context=context)
        except ValueError as exc:
            report.add_error(str(exc), context=context)


def _validate_prompt_files(report: ValidationReport, folder: Path, name: str, config: Mapping[str, Any]) -> None:
    if config.get("prompt_pack") or config.get("prompt_system") or config.get("prompt_template"):
        return
    system_path = folder / "system_prompt.md"
    user_path = folder / "user_prompt.md"
    if not system_path.exists() or not system_path.read_text(encoding="utf-8").strip():
        report.add_error("Missing or empty system prompt", context=f"experiment:{name}")
    if not user_path.exists() or not user_path.read_text(encoding="utf-8").strip():
        report.add_error("Missing or empty user prompt", context=f"experiment:{name}")


def _find_duplicates(items: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return [item for item, count in counts.items() if count > 1]


__all__ = [
    "ConfigurationError",
    "SuiteValidationReport",
    "ValidationMessage",
    "ValidationReport",
    "validate_schema",
    "validate_settings",
    "validate_suite",
]


_PLUGIN_REFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "plugin": {"type": "string"},
        "options": {"type": "object"},
    },
    "required": ["plugin"],
    "additionalProperties": True,
}

_MIDDLEWARE_DEF_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "plugin": {"type": "string"},
        "options": {"type": "object"},
    },
    "anyOf": [
        {"required": ["name"]},
        {"required": ["plugin"]},
    ],
    "additionalProperties": True,
}

_PLUGIN_DEF_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "options": {"type": "object"},
    },
    "required": ["name"],
    "additionalProperties": True,
}

_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "datasource": _PLUGIN_REFERENCE_SCHEMA,
        "llm": _PLUGIN_REFERENCE_SCHEMA,
        "sinks": {
            "type": "array",
            "items": _PLUGIN_REFERENCE_SCHEMA,
            "minItems": 1,
        },
        "prompt_packs": {"type": "object"},
        "suite_defaults": {"type": "object"},
        "retry": {"type": "object"},
        "checkpoint": {"type": "object"},
        "concurrency": {"type": "object"},
        "early_stop": {"type": "object"},
        "early_stop_plugins": {"type": "array", "items": _PLUGIN_DEF_SCHEMA},
    },
    "required": ["datasource", "sinks"],
    "additionalProperties": True,
}

_EXPERIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "temperature": {"type": "number"},
        "max_tokens": {"type": "integer", "minimum": 1},
        "enabled": {"type": "boolean"},
        "is_baseline": {"type": "boolean"},
        "prompt_pack": {"type": "string"},
        "criteria": {"type": "array"},
        "row_plugins": {"type": "array", "items": _PLUGIN_DEF_SCHEMA},
        "aggregator_plugins": {"type": "array", "items": _PLUGIN_DEF_SCHEMA},
        "baseline_plugins": {"type": "array", "items": _PLUGIN_DEF_SCHEMA},
        "llm_middlewares": {"type": "array", "items": _MIDDLEWARE_DEF_SCHEMA},
        "sinks": {"type": "array", "items": _PLUGIN_REFERENCE_SCHEMA},
        "rate_limiter": {"type": "object"},
        "cost_tracker": {"type": "object"},
        "prompt_defaults": {"type": "object"},
        "concurrency": {"type": "object"},
        "early_stop_plugins": {"type": "array", "items": _PLUGIN_DEF_SCHEMA},
    },
    "required": ["temperature", "max_tokens"],
    "additionalProperties": True,
}


@dataclass
class SuiteValidationReport:
    report: ValidationReport
    preflight: dict[str, Any] = field(default_factory=dict)

    def raise_if_errors(self) -> None:
        self.report.raise_if_errors()
