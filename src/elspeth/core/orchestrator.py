"""SDA (Sense/Decide/Act) orchestrator coordinating data input, decision-making, and action execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.core.sda.plugin_registry import create_aggregation_transform, create_halt_condition_plugin, create_transform_plugin
from elspeth.core.sda.runner import SDARunner

if TYPE_CHECKING:
    from elspeth.core.controls import CostTracker, RateLimiter
    from elspeth.core.interfaces import DataSource, LLMClientProtocol, ResultSink


@dataclass
class SDAConfig:
    llm_prompt: dict[str, str] = field(default_factory=dict)
    prompt_fields: list[str] | None = None
    prompt_aliases: dict[str, str] | None = None
    criteria: list[dict[str, str]] | None = None
    transform_plugin_defs: list[dict[str, Any]] | None = None
    aggregation_transform_defs: list[dict[str, Any]] | None = None
    sink_defs: list[dict[str, Any]] | None = None
    prompt_pack: str | None = None
    baseline_plugin_defs: list[dict[str, Any]] | None = None
    retry_config: dict[str, Any] | None = None
    checkpoint_config: dict[str, Any] | None = None
    llm_middleware_defs: list[dict[str, Any]] | None = None
    prompt_defaults: dict[str, Any] | None = None
    concurrency_config: dict[str, Any] | None = None
    halt_condition_config: dict[str, Any] | None = None
    halt_condition_plugin_defs: list[dict[str, Any]] | None = None
    config_fingerprint: str | None = None


class SDAOrchestrator:
    def __init__(
        self,
        *,
        datasource: DataSource,
        sinks: list[ResultSink],
        config: SDAConfig,
        llm_client: LLMClientProtocol | None = None,
        sda_runner: SDARunner | None = None,
        rate_limiter: RateLimiter | None = None,
        cost_tracker: CostTracker | None = None,
        name: str = "default",
        row_limit: int | None = None,
    ):
        self.datasource = datasource
        self.llm_client = llm_client
        self.sinks = sinks
        self.config = config
        self.rate_limiter = rate_limiter
        self.cost_tracker = cost_tracker
        self.name = name
        self.row_limit = row_limit
        transform_plugins = None
        if config.transform_plugin_defs:
            transform_plugins = [create_transform_plugin(defn) for defn in config.transform_plugin_defs]
        aggregation_transforms = None
        if config.aggregation_transform_defs:
            aggregation_transforms = [create_aggregation_transform(defn) for defn in config.aggregation_transform_defs]
        halt_condition_plugins = None
        if config.halt_condition_plugin_defs:
            halt_condition_plugins = [create_halt_condition_plugin(defn) for defn in config.halt_condition_plugin_defs]
        self.halt_condition_plugins = halt_condition_plugins

        self.sda_runner = sda_runner or SDARunner(
            sinks=sinks,
            transform_plugins=transform_plugins or [],
            aggregation_transforms=aggregation_transforms,
            cycle_name=name,
            checkpoint_config=config.checkpoint_config,
            halt_condition_plugins=halt_condition_plugins,
            halt_condition_config=config.halt_condition_config,
            concurrency_config=config.concurrency_config,
            prompt_fields=config.prompt_fields,
            prompt_aliases=config.prompt_aliases,
            config_fingerprint=config.config_fingerprint,
        )

    def run(self) -> dict[str, Any]:
        df = self.datasource.load()
        if self.row_limit is not None and self.row_limit > 0:
            df = df.head(self.row_limit)

        payload = self.sda_runner.run(df)
        return payload
