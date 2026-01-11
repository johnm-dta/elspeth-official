"""Config loader for orchestrator settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from elspeth.core.config_merger import ConfigSource, ConfigurationMerger
from elspeth.core.controls import create_cost_tracker, create_rate_limiter
from elspeth.core.orchestrator import SDAConfig
from elspeth.core.prompts.resolvers import is_remote_pack_url, resolve_remote_pack
from elspeth.core.registry import registry
from elspeth.core.sda.fingerprint import compute_config_fingerprint
from elspeth.core.sda.plugin_registry import normalize_halt_condition_definitions
from elspeth.core.secrets import load_secrets, substitute_variables


@dataclass
class Settings:
    datasource: Any
    sinks: Any
    orchestrator_config: SDAConfig
    llm: Any | None = None
    orchestrator_type: str = "experimental"  # "standard" or "experimental"
    suite_root: Path | None = None
    suite_defaults: dict[str, Any] = field(default_factory=dict)
    rate_limiter: Any | None = None
    cost_tracker: Any | None = None
    prompt_packs: dict[str, Any] = field(default_factory=dict)
    prompt_pack: str | None = None
    landscape_config: dict[str, Any] = field(default_factory=dict)
    source_paths: list[Path] = field(default_factory=list)


def load_settings(
    path: str | Path,
    profile: str = "default",
    secrets_path: Path | None = None,
) -> Settings:
    """Load settings from YAML configuration file.

    Uses ConfigurationMerger for consistent precedence:
    1. System defaults (if any)
    2. Prompt pack
    3. Profile
    4. Suite defaults
    5. Experiment config (handled by suite_runner)

    Args:
        path: Path to settings YAML file
        profile: Profile name to load (default: "default")
        secrets_path: Optional path to secrets YAML for variable substitution
    """
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    # Track source file paths for fingerprinting
    source_paths: list[Path] = [config_path]

    # Apply secrets substitution BEFORE profile extraction
    # This ensures ${VAR} in profile names and all config values get substituted
    if secrets_path:
        secrets = load_secrets(secrets_path)
        data = substitute_variables(data, secrets)

    profile_data = dict(data.get(profile, {}))

    prompt_packs = profile_data.pop("prompt_packs", {})
    prompt_pack_name = profile_data.get("prompt_pack")

    # Check if prompt_pack is a remote URL and resolve it
    pack: dict[str, Any] | None
    if prompt_pack_name and is_remote_pack_url(prompt_pack_name):
        pack = resolve_remote_pack(prompt_pack_name)
        # Store with URL as key for reference
        prompt_packs[prompt_pack_name] = pack
    else:
        pack = prompt_packs.get(prompt_pack_name) if prompt_pack_name else None

    # Use ConfigurationMerger for prompt pack merging
    merger = ConfigurationMerger()

    # Merge pack config into profile config
    if pack:
        pack_source = ConfigSource(name="prompt_pack", data=pack, precedence=2)
        profile_source = ConfigSource(name="profile", data=profile_data, precedence=3)
        merged_config = merger.merge(pack_source, profile_source)
    else:
        merged_config = profile_data

    # Extract merged values
    datasource_cfg = merged_config["datasource"]
    datasource = registry.create_datasource(datasource_cfg["plugin"], datasource_cfg.get("options", {}))

    llm_cfg = merged_config.get("llm")
    llm = registry.create_llm(llm_cfg["plugin"], llm_cfg.get("options", {})) if llm_cfg else None

    # Plugins are now properly appended by merger
    transform_plugin_defs = merged_config.get("row_plugins", [])
    aggregation_transform_defs = merged_config.get("aggregator_plugins", [])
    baseline_plugin_defs = merged_config.get("baseline_plugins", [])
    sink_defs = merged_config.get("sinks", [])
    llm_middleware_defs = merged_config.get("llm_middlewares", [])

    # Other config extraction
    rate_limiter_def = merged_config.get("rate_limiter")
    cost_tracker_def = merged_config.get("cost_tracker")
    prompt_defaults = merged_config.get("prompt_defaults")
    concurrency_config = merged_config.get("concurrency")
    halt_condition_config = merged_config.get("early_stop")
    halt_condition_plugin_defs = normalize_halt_condition_definitions(merged_config.get("early_stop_plugins")) or []

    if not halt_condition_plugin_defs and halt_condition_config:
        halt_condition_plugin_defs = normalize_halt_condition_definitions(halt_condition_config)

    prompts = merged_config.get("prompts", {})
    prompt_fields = merged_config.get("prompt_fields")
    prompt_aliases = merged_config.get("prompt_aliases")
    criteria = merged_config.get("criteria")

    # Create sinks and controls
    sinks = []
    for index, item in enumerate(sink_defs):
        plugin = item["plugin"]
        raw_options = dict(item.get("options", {}))
        artifacts_cfg = raw_options.pop("artifacts", None)
        security_level = raw_options.pop("security_level", item.get("security_level"))
        sink = registry.create_sink(plugin, raw_options)
        setattr(sink, "_dmp_artifact_config", artifacts_cfg or {})
        setattr(sink, "_dmp_plugin_name", plugin)
        base_name = item.get("name") or plugin or f"sink{index}"
        setattr(sink, "_dmp_sink_name", base_name)
        if security_level:
            setattr(sink, "_dmp_security_level", security_level)
        sinks.append(sink)
    rate_limiter = create_rate_limiter(rate_limiter_def)
    cost_tracker = create_cost_tracker(cost_tracker_def)

    # Handle suite_defaults merging with prompt packs
    suite_defaults = dict(merged_config.get("suite_defaults", {}))
    suite_pack_name = suite_defaults.get("prompt_pack")
    if suite_pack_name:
        # Check if suite prompt pack is a remote URL
        if is_remote_pack_url(suite_pack_name):
            suite_pack = resolve_remote_pack(suite_pack_name)
            prompt_packs[suite_pack_name] = suite_pack
        else:
            suite_pack = prompt_packs.get(suite_pack_name)

        if suite_pack:
            suite_pack_source = ConfigSource(name="suite_prompt_pack", data=suite_pack, precedence=2)
            suite_source = ConfigSource(name="suite_defaults", data=suite_defaults, precedence=3)
            suite_defaults = merger.merge(suite_pack_source, suite_source)

    suite_root = merged_config.get("suite_root")
    orchestrator_type = merged_config.get("orchestrator_type", "experimental")
    landscape_config = merged_config.get("landscape", {})

    # Collect prompt file paths from plugins
    for plugin_def in transform_plugin_defs:
        if plugin_def.get("plugin") == "llm_query":
            opts = plugin_def.get("options", {})
            for query in opts.get("queries", []):
                prompt_folder = query.get("prompt_folder")
                if prompt_folder:
                    folder = Path(prompt_folder)
                    for name in ("system.md", "system_prompt.md", "user.md", "user_prompt.md"):
                        prompt_file = folder / name
                        if prompt_file.exists():
                            source_paths.append(prompt_file)

    # Compute config fingerprint from source paths
    config_fingerprint = compute_config_fingerprint(source_paths)

    # Create orchestrator config with fingerprint
    orchestrator_config = SDAConfig(
        llm_prompt=prompts,
        prompt_fields=prompt_fields,
        prompt_aliases=prompt_aliases,
        criteria=criteria,
        transform_plugin_defs=transform_plugin_defs,
        aggregation_transform_defs=aggregation_transform_defs,
        baseline_plugin_defs=baseline_plugin_defs,
        sink_defs=sink_defs,
        prompt_pack=prompt_pack_name,
        retry_config=merged_config.get("retry"),
        checkpoint_config=merged_config.get("checkpoint"),
        llm_middleware_defs=llm_middleware_defs,
        prompt_defaults=prompt_defaults,
        concurrency_config=concurrency_config,
        halt_condition_config=halt_condition_config,
        halt_condition_plugin_defs=halt_condition_plugin_defs or None,
        config_fingerprint=config_fingerprint,
    )

    return Settings(
        datasource=datasource,
        sinks=sinks,
        orchestrator_config=orchestrator_config,
        llm=llm,
        orchestrator_type=orchestrator_type,
        suite_root=Path(suite_root) if suite_root else None,
        suite_defaults=suite_defaults,
        rate_limiter=rate_limiter,
        cost_tracker=cost_tracker,
        prompt_packs=prompt_packs,
        prompt_pack=prompt_pack_name,
        landscape_config=landscape_config,
        source_paths=source_paths,
    )
