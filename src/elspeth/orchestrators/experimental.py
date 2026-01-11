"""Experimental orchestrator for A/B testing with baseline comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from elspeth.core.landscape import RunLandscape, reset_landscape, set_current_landscape
from elspeth.core.sda.plugin_registry import create_baseline_plugin

from .standard import StandardOrchestrator

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from elspeth.core.interfaces import ResultSink
    from elspeth.core.sda.config import SDACycleConfig


@dataclass
class ExperimentalOrchestrator(StandardOrchestrator):
    """Experimental orchestrator with A/B testing and baseline comparison.

    Extends StandardOrchestrator with:
    - Baseline identification from metadata
    - Runs baseline first (before variants)
    - Applies comparison plugins between baseline and variants
    - Includes comparison results in output

    Note: LLM configuration is now handled within transform plugins
    (like llm_query). The orchestrator only builds transform plugins
    from row_plugins config.
    """

    def _identify_baseline(self) -> SDACycleConfig | None:
        """Identify baseline cycle from metadata.

        Looks for cycle with metadata.is_baseline = true.
        Falls back to first cycle if no baseline specified.
        """
        # Check metadata first
        for cycle in self.suite.cycles:
            if cycle.metadata.get("is_baseline"):
                return cycle

        # Fall back to first cycle
        return self.suite.cycles[0] if self.suite.cycles else None

    def _get_variants(self, baseline: SDACycleConfig) -> list[SDACycleConfig]:
        """Get all cycles except baseline."""
        return [c for c in self.suite.cycles if c != baseline]

    def run(
        self,
        df,
        defaults: dict[str, Any] | None = None,
        sink_factory: Callable[[SDACycleConfig], list[ResultSink]] | None = None,
        preflight_info: dict[str, Any] | None = None,
        config_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        """Run suite with baseline-first ordering and comparison logic.

        Execution order:
        1. Identify baseline from metadata
        2. Run baseline first
        3. Run variants
        4. Apply comparison plugins (baseline vs each variant)
        5. Include comparison results in output
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
            return self._run_experiments(df, defaults, prompt_packs, sink_factory, preflight_info)
        finally:
            # Write manifest before cleanup
            landscape.write_manifest()
            reset_landscape(token)
            landscape.cleanup()

    def _run_experiments(
        self,
        df,
        defaults: dict[str, Any],
        prompt_packs: dict[str, Any],
        sink_factory: Callable[[SDACycleConfig], list[ResultSink]] | None,
        preflight_info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Execute experiments with landscape context active."""
        results: dict[str, Any] = {}

        # Identify baseline and order experiments: baseline first, then variants
        baseline = self._identify_baseline()
        if not baseline:
            # No cycles to run
            return results

        experiments: list[SDACycleConfig] = [baseline]
        experiments.extend(self._get_variants(baseline))

        # Build metadata with is_baseline flag (reserved for future landscape use)
        _suite_metadata = [
            {
                "experiment": exp.name,
                "temperature": exp.temperature,
                "max_tokens": exp.max_tokens,
                "is_baseline": (exp == baseline),
            }
            for exp in experiments
        ]

        if preflight_info is None:
            preflight_info = {
                "experiment_count": len(experiments),
                "baseline": baseline.name,
            }

        baseline_payload = None

        for experiment in experiments:
            pack_name = experiment.prompt_pack or defaults.get("prompt_pack")
            pack = prompt_packs.get(pack_name) if pack_name else None

            # Determine sinks
            sinks = self._select_sinks(experiment, defaults, pack, sink_factory)

            # Build and run
            runner = self.build_runner(
                experiment,
                {**defaults, "prompt_packs": prompt_packs, "prompt_pack": pack_name},
                sinks,
            )

            # Execute cycle
            payload = runner.run(df)

            # Track baseline payload
            if experiment == baseline:
                baseline_payload = payload

            # Store results
            results[experiment.name] = {
                "payload": payload,
                "config": experiment,
            }

            # Apply baseline comparison for variants
            if baseline_payload and experiment != baseline:
                comparisons = self._compare_with_baseline(
                    baseline_payload,
                    payload,
                    experiment,
                    defaults,
                    pack,
                )
                if comparisons:
                    payload["baseline_comparison"] = comparisons
                    results[experiment.name]["baseline_comparison"] = comparisons

        return results

    def _compare_with_baseline(
        self,
        baseline_payload: dict[str, Any],
        variant_payload: dict[str, Any],
        experiment: SDACycleConfig,
        defaults: dict[str, Any],
        pack: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Apply comparison plugins between baseline and variant.

        Comparison plugin precedence:
        1. defaults.baseline_plugin_defs
        2. pack.baseline_plugins (appended)
        3. experiment.metadata.baseline_plugins (appended)
        """
        # Collect comparison plugin definitions
        comp_defs = list(defaults.get("baseline_plugin_defs", []))

        if pack and pack.get("baseline_plugins"):
            comp_defs = list(pack.get("baseline_plugins", [])) + comp_defs

        # Check experiment metadata for baseline plugins
        if experiment.metadata.get("baseline_plugins"):
            comp_defs += experiment.metadata["baseline_plugins"]

        # Execute comparison plugins
        comparisons = {}
        for defn in comp_defs:
            plugin = create_baseline_plugin(defn)
            diff = plugin.compare(baseline_payload, variant_payload)
            if diff:
                comparisons[plugin.name] = diff

        return comparisons
