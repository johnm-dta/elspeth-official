"""Integration tests for refactored SDA components."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from elspeth.core.sda.runner import SDARunner


class MockSink:
    """Mock sink for testing."""

    def __init__(self):
        self.written_payload: dict[str, Any] | None = None
        self.written_metadata: dict[str, Any] | None = None

    def write(self, payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        self.written_payload = payload
        self.written_metadata = metadata

    def produces(self) -> list[str]:
        return []

    def consumes(self) -> list[str]:
        return []

    def finalize(self, artifacts: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> None:
        pass


class MockTransformPlugin:
    """Mock transform plugin for testing."""

    name = "mock_transform"

    def __init__(self, process_fn=None):
        self._process_fn = process_fn or (lambda row, ctx: row)
        self.calls: list[tuple[dict, dict]] = []

    def transform(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((dict(row), dict(context)))
        return self._process_fn(row, context)


def test_full_sda_pipeline_with_all_features(tmp_path):
    """Full SDA pipeline with checkpoints, transforms, plugins."""
    checkpoint_path = tmp_path / "checkpoint.jsonl"

    # Sample data
    df = pd.DataFrame(
        [
            {"id": "1", "text": "First"},
            {"id": "2", "text": "Second"},
            {"id": "3", "text": "Third"},
        ]
    )

    # Mock components
    sink = MockSink()

    def add_processed_flag(row, ctx):
        row["processed"] = True
        return row

    plugin = MockTransformPlugin(add_processed_flag)

    # Run with all features
    runner = SDARunner(
        sinks=[sink],
        transform_plugins=[plugin],
        prompt_fields=["text", "id"],  # Include id for checkpoint
        checkpoint_config={"path": str(checkpoint_path), "field": "id"},
    )

    result = runner.run(df)

    # Verify all rows processed
    assert len(result["results"]) == 3
    assert result["metadata"]["rows"] == 3
    assert result["metadata"]["row_count"] == 3

    # Verify checkpoint created (should have 3 entries written as JSONL)
    assert checkpoint_path.exists(), "Checkpoint file should be created"
    checkpoint_lines = checkpoint_path.read_text().strip().split("\n")
    assert len(checkpoint_lines) == 3, f"Expected 3 checkpoint entries, got {len(checkpoint_lines)}"
    checkpoint_ids = {json.loads(line)["_checkpoint_id"] for line in checkpoint_lines}
    assert checkpoint_ids == {"1", "2", "3"}, f"Expected IDs 1,2,3, got {checkpoint_ids}"

    # Verify sink received payload
    assert sink.written_payload is not None
    assert len(sink.written_payload["results"]) == 3

    # Verify plugin was called for each row
    assert len(plugin.calls) == 3

    # Verify rows have processed flag
    for record in result["results"]:
        assert record["row"]["processed"] is True


def test_refactored_components_work_together():
    """Verify all refactored components integrate correctly."""
    # This test exists to ensure no regressions in component integration
    df = pd.DataFrame([{"id": "1", "text": "Test"}])

    plugin = MockTransformPlugin()

    runner = SDARunner(
        sinks=[MockSink()],
        transform_plugins=[plugin],
        prompt_fields=["text", "id"],
    )

    result = runner.run(df)

    # Should work with transform plugins
    assert len(result["results"]) == 1
    assert "metadata" in result
    assert result["metadata"]["rows"] == 1

    # Plugin should have been called
    assert len(plugin.calls) == 1


def test_sda_pipeline_with_context_communication():
    """Test that context dict enables inter-plugin communication."""
    df = pd.DataFrame([{"id": "1", "value": "100"}])

    def step1_transform(row, ctx):
        ctx["step1_complete"] = True
        ctx["computed_value"] = int(row.get("value", 0)) * 2
        return row

    def step2_transform(row, ctx):
        # Can read context from step1
        assert ctx.get("step1_complete") is True
        row["doubled_value"] = ctx.get("computed_value", 0)
        return row

    plugin1 = MockTransformPlugin(step1_transform)
    plugin2 = MockTransformPlugin(step2_transform)

    runner = SDARunner(
        sinks=[MockSink()],
        transform_plugins=[plugin1, plugin2],
        prompt_fields=["id", "value"],
    )

    result = runner.run(df)

    # Both plugins should have been called
    assert len(plugin1.calls) == 1
    assert len(plugin2.calls) == 1

    # Result should have computed value from step2
    assert result["results"][0]["row"]["doubled_value"] == 200
