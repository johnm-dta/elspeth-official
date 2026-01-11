"""SDA (Sense/Decide/Act) runner executing one complete orchestration cycle."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elspeth.core.artifact_pipeline import ArtifactPipeline, SinkBinding
from elspeth.core.processing import prepare_prompt_context
from elspeth.core.sda.checkpoint import CheckpointManager
from elspeth.core.sda.early_stop import EarlyStopCoordinator
from elspeth.core.sda.plugin_registry import create_halt_condition_plugin
from elspeth.core.sda.result_aggregator import ResultAggregator
from elspeth.core.sda.row_processor import RowProcessor
from elspeth.core.security import normalize_security_level

if TYPE_CHECKING:
    import pandas as pd

    from elspeth.core.interfaces import ResultSink
    from elspeth.core.sda.plugins import AggregationTransform, HaltConditionPlugin, TransformPlugin

logger = logging.getLogger(__name__)


@dataclass
class SDARunner:
    """Executes one complete SDA cycle.

    The runner is now a generic plugin executor - all LLM logic lives
    in transform plugins (like llm_query). The runner handles:
    - Row iteration
    - Checkpointing
    - Early stopping
    - Concurrent processing
    - Result aggregation
    - Sink coordination
    """

    sinks: list[ResultSink]
    transform_plugins: list[TransformPlugin] = field(default_factory=list)
    aggregation_transforms: list[AggregationTransform] | None = None
    cycle_name: str | None = None
    checkpoint_config: dict[str, Any] | None = None
    security_level: str | None = None
    halt_condition_plugins: list[HaltConditionPlugin] | None = None
    halt_condition_config: dict[str, Any] | None = None
    concurrency_config: dict[str, Any] | None = None
    prompt_fields: list[str] | None = None
    prompt_aliases: dict[str, str] | None = None
    config_fingerprint: str | None = None
    _active_security_level: str | None = field(default=None, init=False, repr=False)

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Execute SDA cycle on DataFrame.

        Args:
            df: Input DataFrame to process

        Returns:
            Payload dict with results, failures, and metadata
        """
        # Initialize early stop coordinator
        plugins = []
        if self.halt_condition_plugins:
            plugins = list(self.halt_condition_plugins)
        elif self.halt_condition_config:
            definition = {"name": "threshold", "options": dict(self.halt_condition_config)}
            plugin = create_halt_condition_plugin(definition)
            plugins = [plugin]
        self._early_stop_coordinator = EarlyStopCoordinator(plugins=plugins)

        # Resolve active security level
        active_level: str | None = None
        if self.security_level is not None:
            active_level = normalize_security_level(self.security_level)
        elif hasattr(df, "attrs"):
            attr_level = df.attrs.get("security_level")
            if attr_level:
                active_level = normalize_security_level(attr_level)
        self._active_security_level = active_level

        # Setup checkpointing
        checkpoint_manager: CheckpointManager | None = None
        checkpoint_field: str | None = None
        if self.checkpoint_config:
            checkpoint_path = Path(self.checkpoint_config.get("path", "checkpoint.jsonl"))
            checkpoint_field = self.checkpoint_config.get("field", "APPID")
            save_results = self.checkpoint_config.get("save_results", True)
            checkpoint_manager = CheckpointManager(
                checkpoint_path,
                checkpoint_field,
                save_results=save_results,
                config_fingerprint=self.config_fingerprint,
            )

        # Create row processor (simplified - just plugins and security)
        row_processor = RowProcessor(
            transform_plugins=self.transform_plugins,
            security_level=self._active_security_level,
        )

        # Create result aggregator
        aggregator = ResultAggregator(
            aggregation_plugins=self.aggregation_transforms or [],
            cost_tracker=None,  # Cost tracking now in llm_query plugin
        )

        # Recover saved results from checkpoint
        if checkpoint_manager and checkpoint_manager.has_saved_results:
            for saved_result in checkpoint_manager.get_saved_results():
                aggregator.add_result(saved_result, row_index=-1)  # -1 indicates recovered
            logger.info("Recovered %d results from checkpoint", checkpoint_manager.processed_count)

        # Prepare rows for processing
        total_rows = len(df)
        checkpointed_count = checkpoint_manager.processed_count if checkpoint_manager else 0
        rows_to_process: list[tuple[int, pd.Series, dict[str, Any], str | None]] = []
        for idx, (_, row) in enumerate(df.iterrows()):
            context = prepare_prompt_context(
                row,
                include_fields=self.prompt_fields,
                alias_map=self.prompt_aliases,
            )
            row_id = context.get(checkpoint_field) if checkpoint_field else None
            if checkpoint_manager and row_id and checkpoint_manager.is_processed(str(row_id)):
                continue
            if self._early_stop_coordinator.is_stopped():
                break
            rows_to_process.append((idx, row, context, row_id))

        remaining_count = len(rows_to_process)
        logger.info(
            "Batch: %d total rows, %d checkpointed, %d remaining",
            total_rows, checkpointed_count, remaining_count
        )

        def handle_success(idx: int, record: dict[str, Any], row_id: str | None) -> None:
            aggregator.add_result(record, row_index=idx)
            if checkpoint_manager and row_id:
                checkpoint_manager.mark_processed(str(row_id), result=record)
            self._early_stop_coordinator.check_record(record, row_index=idx)

        def handle_failure(failure: dict[str, Any]) -> None:
            aggregator.add_failure(failure)

        # Process rows (sequential or parallel)
        concurrency_cfg = self.concurrency_config or {}
        if rows_to_process and self._should_run_parallel(concurrency_cfg, len(rows_to_process)):
            self._run_parallel(
                rows_to_process,
                row_processor,
                handle_success,
                handle_failure,
                concurrency_cfg,
            )
        else:
            for progress_idx, (idx, row, row_data, row_id) in enumerate(rows_to_process, start=1):
                if self._early_stop_coordinator.is_stopped():
                    break
                logger.info(
                    "Processing row %d/%d (ID: %s)",
                    progress_idx, remaining_count, row_id or f"row_{idx}"
                )
                record, failure = row_processor.process_row(row, row_data, row_id)
                if record:
                    handle_success(idx, record, row_id)
                if failure:
                    handle_failure(failure)

        # Build payload with aggregator
        payload = aggregator.build_payload(
            security_level=self._active_security_level,
            early_stop_reason=self._early_stop_coordinator.get_reason(),
        )

        # Execute sink pipeline
        pipeline = ArtifactPipeline(self._build_sink_bindings())
        pipeline.execute(payload, payload["metadata"])
        self._active_security_level = None
        return payload

    def _should_run_parallel(self, config: dict[str, Any], backlog_size: int) -> bool:
        """Check if parallel processing should be used."""
        if not config or not config.get("enabled"):
            return False
        max_workers = max(int(config.get("max_workers", 1)), 1)
        if max_workers <= 1:
            return False
        threshold = int(config.get("backlog_threshold", 50))
        return backlog_size >= threshold

    def _run_parallel(
        self,
        rows_to_process: list[tuple[int, pd.Series, dict[str, Any], str | None]],
        row_processor: RowProcessor,
        handle_success,
        handle_failure,
        config: dict[str, Any],
    ) -> None:
        """Run row processing in parallel."""
        max_workers = max(int(config.get("max_workers", 4)), 1)
        lock = threading.Lock()
        total = len(rows_to_process)
        completed = [0]  # Use list to allow mutation in closure

        def worker(data: tuple[int, pd.Series, dict[str, Any], str | None]) -> None:
            if self._early_stop_coordinator.is_stopped():
                return
            idx, row, row_data, row_id = data
            record, failure = row_processor.process_row(row, row_data, row_id)
            with lock:
                completed[0] += 1
                logger.info(
                    "Completed row %d/%d (ID: %s)",
                    completed[0], total, row_id or f"row_{idx}"
                )
                if record:
                    handle_success(idx, record, row_id)
                if failure:
                    handle_failure(failure)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for data in rows_to_process:
                if self._early_stop_coordinator.is_stopped():
                    break
                executor.submit(worker, data)

    def _build_sink_bindings(self) -> list[SinkBinding]:
        """Build sink bindings for artifact pipeline."""
        bindings: list[SinkBinding] = []
        for index, sink in enumerate(self.sinks):
            artifact_config = getattr(sink, "_dmp_artifact_config", {}) or {}
            plugin = getattr(sink, "_dmp_plugin_name", sink.__class__.__name__)
            base_id = getattr(sink, "_dmp_sink_name", plugin)
            sink_id = f"{base_id}:{index}"
            security_level = getattr(sink, "_dmp_security_level", None)
            if security_level is not None:
                security_level = normalize_security_level(security_level)
            bindings.append(
                SinkBinding(
                    id=sink_id,
                    plugin=plugin,
                    sink=sink,
                    artifact_config=artifact_config,
                    original_index=index,
                    security_level=security_level,
                )
            )
        return bindings
