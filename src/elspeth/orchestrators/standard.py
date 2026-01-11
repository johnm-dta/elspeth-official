"""Standard orchestrator for sequential SDA cycle execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from elspeth.core import registry as core_registry
from elspeth.core.config_merger import ConfigSource, ConfigurationMerger
from elspeth.core.landscape import RunLandscape, reset_landscape, set_current_landscape
from elspeth.core.sda.plugin_registry import (
    create_aggregation_transform,
    create_halt_condition_plugin,
    create_transform_plugin,
    normalize_halt_condition_definitions,
)
from elspeth.core.sda.runner import SDARunner
from elspeth.core.security import resolve_security_level

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from elspeth.core.interfaces import ResultSink
    from elspeth.core.sda.config import SDACycleConfig, SDASuite


@dataclass
class StandardOrchestrator:
    """Standard orchestrator - executes SDA cycles sequentially.

    No baseline tracking, no comparison logic.
    Just runs cycles in order as defined in suite.

    Note: LLM configuration is now handled within transform plugins
    (like llm_query). The orchestrator only builds transform plugins
    from row_plugins config.
    """

    suite: SDASuite
    sinks: list[ResultSink]

    def build_runner(
        self,
        config: SDACycleConfig,
        defaults: dict[str, Any],
        sinks: list[ResultSink],
    ) -> SDARunner:
        """Build runner for single cycle with merged configuration.

        Merging precedence:
        1. defaults (from Settings)
        2. prompt_pack (if specified)
        3. config (cycle-specific)
        """
        merger = ConfigurationMerger()

        # Build sources in precedence order
        sources = []

        # Source 1: defaults
        sources.append(ConfigSource(name="defaults", data=defaults, precedence=1))

        # Source 2: prompt pack (if specified)
        prompt_packs = defaults.get("prompt_packs", {})
        pack_name = config.prompt_pack or defaults.get("prompt_pack")
        if pack_name and pack_name in prompt_packs:
            pack = prompt_packs[pack_name]
            sources.append(ConfigSource(name="prompt_pack", data=pack, precedence=2))

        # Source 3: cycle config
        config_data = {k: v for k, v in config.__dict__.items() if v is not None and not k.startswith("_")}
        sources.append(ConfigSource(name="cycle", data=config_data, precedence=3))

        # Merge all sources
        merged = merger.merge(*sources)

        # Extract merged values
        prompt_fields = merged.get("prompt_fields")
        prompt_aliases = merged.get("prompt_aliases")
        concurrency_config = merged.get("concurrency_config") or merged.get("concurrency")
        halt_condition_config = merged.get("halt_condition_config") or merged.get("early_stop")

        # Handle halt condition plugins (appended by merger)
        halt_condition_plugin_defs = merged.get("halt_condition_plugin_defs", []) or merged.get("halt_condition_plugins", [])
        if halt_condition_plugin_defs:
            halt_condition_plugin_defs = normalize_halt_condition_definitions(halt_condition_plugin_defs)
        if not halt_condition_plugin_defs and halt_condition_config:
            halt_condition_plugin_defs = normalize_halt_condition_definitions(halt_condition_config)
        halt_condition_plugins = (
            [create_halt_condition_plugin(defn) for defn in halt_condition_plugin_defs] if halt_condition_plugin_defs else None
        )

        # Security level resolution
        pack = prompt_packs.get(pack_name) if pack_name and pack_name in prompt_packs else None
        security_level = resolve_security_level(
            config.security_level,
            (pack.get("security_level") if pack else None),
            defaults.get("security_level"),
        )

        # Transform plugins (appended by merger)
        # Now the primary way to configure LLM queries
        transform_plugin_defs = (
            merged.get("transform_plugin_defs", []) or merged.get("row_plugins", []) or merged.get("transform_plugins", [])
        )
        transform_plugins = [create_transform_plugin(defn) for defn in transform_plugin_defs] if transform_plugin_defs else []

        # Aggregation transforms (appended by merger)
        aggregation_transform_defs = (
            merged.get("aggregation_transform_defs", []) or merged.get("aggregator_plugins", []) or merged.get("aggregation_transforms", [])
        )
        aggregation_transforms = (
            [create_aggregation_transform(defn) for defn in aggregation_transform_defs] if aggregation_transform_defs else None
        )

        return SDARunner(
            sinks=sinks,
            transform_plugins=transform_plugins,
            aggregation_transforms=aggregation_transforms,
            cycle_name=config.name,
            prompt_fields=prompt_fields,
            prompt_aliases=prompt_aliases,
            concurrency_config=concurrency_config,
            security_level=security_level,
            halt_condition_plugins=halt_condition_plugins,
            halt_condition_config=halt_condition_config,
        )

    def _instantiate_sinks(self, defs: list[dict[str, Any]]) -> list[ResultSink]:
        sinks: list[ResultSink] = []
        for index, entry in enumerate(defs):
            plugin = entry.get("plugin")
            if not isinstance(plugin, str):
                raise ValueError(f"Sink definition at index {index} missing 'plugin' key")
            raw_options = dict(entry.get("options", {}))
            core_registry.registry.validate_sink(plugin, raw_options)
            options = dict(raw_options)
            artifacts_cfg = options.pop("artifacts", None)
            security_level = options.pop("security_level", entry.get("security_level"))
            sink = core_registry.registry.create_sink(plugin, options)
            setattr(sink, "_dmp_artifact_config", artifacts_cfg or {})
            setattr(sink, "_dmp_plugin_name", plugin)
            base_name = entry.get("name") or plugin or f"sink{index}"
            setattr(sink, "_dmp_sink_name", base_name)
            if security_level:
                setattr(sink, "_dmp_security_level", security_level)
            sinks.append(sink)
        return sinks

    def _select_sinks(
        self,
        cycle: SDACycleConfig,
        defaults: dict[str, Any],
        pack: dict[str, Any] | None,
        sink_factory: Callable[[SDACycleConfig], list[ResultSink]] | None,
    ) -> list[ResultSink]:
        """Resolve sinks for a cycle using cycle/pack/defaults/sink_factory in order."""
        if cycle.sink_defs:
            return self._instantiate_sinks(cycle.sink_defs)
        if pack and pack.get("sinks"):
            return self._instantiate_sinks(pack["sinks"])
        if defaults.get("sink_defs"):
            return self._instantiate_sinks(defaults["sink_defs"])
        return sink_factory(cycle) if sink_factory else self.sinks

    def run(
        self,
        df,
        defaults: dict[str, Any] | None = None,
        sink_factory: Callable[[SDACycleConfig], list[ResultSink]] | None = None,
        preflight_info: dict[str, Any] | None = None,
        config_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        """Run all cycles in suite sequentially.

        No baseline tracking, no comparison logic.
        Returns results dict keyed by cycle name.
        """
        defaults = defaults or {}
        prompt_packs = defaults.get("prompt_packs", {})

        # Extract landscape config and create landscape
        landscape_config = defaults.get("landscape", {})
        landscape = RunLandscape(
            base_path=landscape_config.get("path"),
            persist=landscape_config.get("persist", False),
            capture_llm_calls=landscape_config.get("capture_llm_calls", True),
            clean_before_run=landscape_config.get("clean_before_run", False),
        )

        # Save config at start
        landscape.save_config(
            original_paths=config_paths or [],
            resolved=defaults,
        )

        # Set landscape context
        token = set_current_landscape(landscape)

        try:
            return self._run_cycles(df, defaults, prompt_packs, sink_factory, preflight_info, landscape)
        finally:
            # Write manifest before cleanup
            landscape.write_manifest()
            reset_landscape(token)
            landscape.cleanup()

    def _run_cycles(
        self,
        df,
        defaults: dict[str, Any],
        prompt_packs: dict[str, Any],
        sink_factory: Callable[[SDACycleConfig], list[ResultSink]] | None,
        preflight_info: dict[str, Any] | None,
        landscape: RunLandscape,
    ) -> dict[str, Any]:
        """Execute cycles with landscape context active."""
        results: dict[str, Any] = {}

        # Build cycle metadata (reserved for future landscape use)
        _cycle_metadata = [
            {
                "cycle": cycle.name,
                "temperature": cycle.temperature,
                "max_tokens": cycle.max_tokens,
            }
            for cycle in self.suite.cycles
        ]

        if preflight_info is None:
            preflight_info = {
                "cycle_count": len(self.suite.cycles),
            }

        # Run cycles in order
        for cycle in self.suite.cycles:
            pack_name = cycle.prompt_pack or defaults.get("prompt_pack")
            pack = prompt_packs.get(pack_name) if pack_name else None

            # Determine sinks
            sinks = self._select_sinks(cycle, defaults, pack, sink_factory)

            # Build and run
            runner = self.build_runner(
                cycle,
                {**defaults, "prompt_packs": prompt_packs, "prompt_pack": pack_name},
                sinks,
            )

            # Execute cycle
            payload = runner.run(df)

            # Store results
            results[cycle.name] = {
                "payload": payload,
                "config": cycle,
            }

        return results
