"""Configuration models for SDA cycles and suites."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.core.config_schema import validate_experiment_config
from elspeth.core.sda.plugin_registry import normalize_halt_condition_definitions
from elspeth.core.validation import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class SDACycleConfig:
    """Configuration for a single SDA cycle.

    Core SDA orchestration parameters only.
    Experiment-specific fields (is_baseline, hypothesis, baseline_plugins)
    should be placed in metadata dict for use by ExperimentalOrchestrator.
    """

    name: str
    temperature: float
    max_tokens: int
    enabled: bool = True
    description: str = ""
    author: str = "unknown"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)  # For experiment-specific data
    options: dict[str, Any] = field(default_factory=dict)
    prompt_system: str = ""
    prompt_template: str = ""
    prompt_fields: list[str] | None = None
    criteria: list[dict[str, Any]] | None = None
    transform_plugin_defs: list[dict[str, Any]] = field(default_factory=list)
    aggregation_transform_defs: list[dict[str, Any]] = field(default_factory=list)
    sink_defs: list[dict[str, Any]] = field(default_factory=list)
    rate_limiter_def: dict[str, Any] | None = None
    cost_tracker_def: dict[str, Any] | None = None
    prompt_pack: str | None = None
    prompt_defaults: dict[str, Any] | None = None
    llm_middleware_defs: list[dict[str, Any]] = field(default_factory=list)
    concurrency_config: dict[str, Any] | None = None
    security_level: str | None = None
    halt_condition_plugin_defs: list[dict[str, Any]] = field(default_factory=list)
    halt_condition_config: dict[str, Any] | None = None

    @classmethod
    def from_file(cls, path: Path) -> SDACycleConfig:
        config_path = path
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            validate_experiment_config(data)
        except ConfigurationError as exc:
            raise ValueError(f"Invalid experiment config '{config_path}': {exc}") from exc
        folder = path.parent

        prompt_system = data.get("prompt_system")
        if not prompt_system:
            system_path = folder / "system_prompt.md"
            prompt_system = system_path.read_text(encoding="utf-8") if system_path.exists() else ""

        prompt_template = data.get("prompt_template")
        if not prompt_template:
            user_path = folder / "user_prompt.md"
            prompt_template = user_path.read_text(encoding="utf-8") if user_path.exists() else ""

        halt_condition_plugin_defs = normalize_halt_condition_definitions(data.get("halt_condition_plugins")) or []
        if not halt_condition_plugin_defs and data.get("early_stop"):
            halt_condition_plugin_defs = normalize_halt_condition_definitions(data.get("early_stop"))

        # Build metadata dict (for experiment-specific fields)
        metadata = data.get("metadata", {})
        # Migrate legacy fields to metadata
        if "is_baseline" in data:
            metadata["is_baseline"] = data["is_baseline"]
        if "hypothesis" in data:
            metadata["hypothesis"] = data["hypothesis"]
        if "baseline_plugins" in data:
            metadata["baseline_plugins"] = data["baseline_plugins"]

        return cls(
            name=data.get("name", path.parent.name),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 512),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            author=data.get("author", "unknown"),
            tags=data.get("tags", []),
            metadata=metadata,
            options=data,
            prompt_system=prompt_system,
            prompt_template=prompt_template,
            prompt_fields=data.get("prompt_fields"),
            criteria=data.get("criteria"),
            transform_plugin_defs=data.get("row_plugins", []),
            aggregation_transform_defs=data.get("aggregator_plugins", []),
            sink_defs=data.get("sinks", []),
            rate_limiter_def=data.get("rate_limiter"),
            cost_tracker_def=data.get("cost_tracker"),
            prompt_pack=data.get("prompt_pack"),
            prompt_defaults=data.get("prompt_defaults"),
            llm_middleware_defs=data.get("llm_middlewares", []),
            concurrency_config=data.get("concurrency"),
            security_level=data.get("security_level"),
            halt_condition_plugin_defs=halt_condition_plugin_defs,
            halt_condition_config=data.get("early_stop"),
        )


@dataclass
class SDASuite:
    """Collection of SDA cycles to execute.

    Orchestrators handle baseline identification from cycle metadata.
    """

    root: Path
    cycles: list[SDACycleConfig]

    @classmethod
    def load(cls, root: Path) -> SDASuite:
        """Load all enabled cycles from directory structure.

        No baseline identification - orchestrators handle that.
        """
        cycles: list[SDACycleConfig] = []

        for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            config_path = folder / "config.json"
            if not config_path.exists():
                continue
            cfg = SDACycleConfig.from_file(config_path)
            if cfg.enabled:
                cycles.append(cfg)

        return cls(root=root, cycles=cycles)
