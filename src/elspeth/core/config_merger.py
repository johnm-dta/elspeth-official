# src/elspeth/core/config_merger.py
"""Centralized configuration merging with documented precedence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar


class MergeStrategy(Enum):
    """Configuration merge strategies."""

    OVERRIDE = "override"  # Higher precedence replaces lower
    APPEND = "append"  # Accumulate from all sources (lists)
    DEEP_MERGE = "deep_merge"  # Recursive merge (nested dicts)


@dataclass
class ConfigSource:
    """Configuration source with metadata."""

    name: str  # "system_default", "prompt_pack", "profile", "suite", "experiment"
    data: dict[str, Any]
    precedence: int  # Lower number = lower precedence


class ConfigurationMerger:
    """Centralized configuration merging with documented precedence.

    Precedence levels (lowest to highest):
    1. System defaults (precedence=1)
    2. Prompt pack (precedence=2)
    3. Profile (precedence=3)
    4. Suite defaults (precedence=4)
    5. Experiment config (precedence=5)

    Merge strategies by key:
    - Scalar values (str, int, bool): OVERRIDE
    - Lists (*_plugins, *_sinks): APPEND
    - Nested dicts (llm.options, datasource.options): DEEP_MERGE
    """

    # Define merge strategies for known keys
    MERGE_STRATEGIES: ClassVar[dict[str, MergeStrategy]] = {
        # Lists use APPEND strategy - add all variants (base and normalized names)
        "row_plugins": MergeStrategy.APPEND,
        "row_plugin_defs": MergeStrategy.APPEND,
        "transform_plugin_defs": MergeStrategy.APPEND,
        "transform_plugins": MergeStrategy.APPEND,
        "aggregator_plugins": MergeStrategy.APPEND,
        "aggregator_plugin_defs": MergeStrategy.APPEND,
        "aggregation_transform_defs": MergeStrategy.APPEND,
        "baseline_plugins": MergeStrategy.APPEND,
        "baseline_plugin_defs": MergeStrategy.APPEND,
        "llm_middlewares": MergeStrategy.APPEND,
        "llm_middleware_defs": MergeStrategy.APPEND,
        "sinks": MergeStrategy.APPEND,
        "sink_defs": MergeStrategy.APPEND,
        "early_stop_plugins": MergeStrategy.APPEND,
        "early_stop_plugin_defs": MergeStrategy.APPEND,
        "halt_condition_plugins": MergeStrategy.APPEND,
        "halt_condition_plugin_defs": MergeStrategy.APPEND,
        # Nested dicts use DEEP_MERGE
        "llm": MergeStrategy.DEEP_MERGE,
        "datasource": MergeStrategy.DEEP_MERGE,
        "orchestrator": MergeStrategy.DEEP_MERGE,
        "concurrency": MergeStrategy.DEEP_MERGE,
        "retry": MergeStrategy.DEEP_MERGE,
        "checkpoint": MergeStrategy.DEEP_MERGE,
        "early_stop": MergeStrategy.DEEP_MERGE,
        "prompts": MergeStrategy.DEEP_MERGE,
        # All other keys: OVERRIDE (default)
    }

    def __init__(self):
        """Initialize merger with empty trace."""
        self._merge_trace: list[dict[str, Any]] = []

    def merge(self, *sources: ConfigSource) -> dict[str, Any]:
        """Merge configuration sources with defined precedence.

        Args:
            sources: Configuration sources in any order (sorted by precedence)

        Returns:
            Merged configuration dictionary
        """
        # Sort by precedence (lowest first)
        sorted_sources = sorted(sources, key=lambda s: s.precedence)

        merged: dict[str, Any] = {}
        self._merge_trace = []  # Reset trace

        for source in sorted_sources:
            merged = self._merge_source(merged, source)

        return merged

    def _merge_source(self, base: dict[str, Any], source: ConfigSource) -> dict[str, Any]:
        """Merge single source into base configuration.

        Args:
            base: Base configuration dictionary
            source: Configuration source to merge

        Returns:
            Merged configuration dictionary
        """
        result = base.copy()

        for key, value in source.data.items():
            strategy = self.MERGE_STRATEGIES.get(key, MergeStrategy.OVERRIDE)

            if strategy == MergeStrategy.OVERRIDE:
                result[key] = value
                self._merge_trace.append({"key": key, "strategy": "override", "source": source.name, "value": value})

            elif strategy == MergeStrategy.APPEND:
                if key not in result:
                    result[key] = []
                # Append new items to existing list
                if isinstance(value, list):
                    result[key] = result[key] + value
                else:
                    result[key].append(value)
                self._merge_trace.append({"key": key, "strategy": "append", "source": source.name, "appended": value})

            elif strategy == MergeStrategy.DEEP_MERGE:
                if key not in result:
                    result[key] = {}
                result[key] = self._deep_merge_dict(result[key], value)
                self._merge_trace.append(
                    {
                        "key": key,
                        "strategy": "deep_merge",
                        "source": source.name,
                        "merged_keys": list(value.keys()) if isinstance(value, dict) else [],
                    }
                )

        return result

    def _deep_merge_dict(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge nested dictionaries.

        Args:
            base: Base dictionary
            override: Dictionary to merge into base

        Returns:
            Merged dictionary (base keys + override keys, override wins on conflicts)
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                # Override scalar values or non-dict types
                result[key] = value
        return result

    def explain(self, key: str, merged_config: dict[str, Any]) -> str:
        """Explain why a configuration value is what it is.

        Args:
            key: Configuration key (e.g., "rate_limit" or "llm.options.temperature")
            merged_config: The merged configuration dictionary

        Returns:
            Explanation string showing source and precedence

        Example:
            >>> merger.explain("rate_limit", config)
            "rate_limit = 50
             Source: profile
             Strategy: override"
        """
        # Handle nested keys (e.g., "llm.options.temperature")
        keys = key.split(".")
        value = merged_config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return f"{key} = <not found>"

        # Find trace entry for this key
        relevant_traces = [t for t in self._merge_trace if t["key"] == keys[0]]

        if not relevant_traces:
            return f"{key} = {value}\nSource: <unknown>"

        # Get the last trace (highest precedence)
        last_trace = relevant_traces[-1]

        explanation = f"{key} = {value}\n"
        explanation += f"Source: {last_trace['source']}\n"
        explanation += f"Strategy: {last_trace['strategy']}"

        return explanation
