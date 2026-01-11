"""Tests for SDARunner (simplified - plugin executor)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

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
    """Mock transform plugin that adds a field."""

    name = "mock_transform"

    def __init__(self, field_name: str = "processed", field_value: Any = True):
        self.field_name = field_name
        self.field_value = field_value
        self.calls: list[tuple[dict, dict]] = []

    def transform(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((dict(row), dict(context)))
        row[self.field_name] = self.field_value
        return row


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        [
            {"id": "1", "text": "Hello world"},
            {"id": "2", "text": "Test data"},
        ]
    )


@pytest.fixture
def mock_sink() -> MockSink:
    """Mock sink fixture."""
    return MockSink()


def test_runner_processes_all_rows(sample_df, mock_sink):
    """SDARunner processes all rows in DataFrame."""
    mock_plugin = MockTransformPlugin()

    runner = SDARunner(
        sinks=[mock_sink],
        transform_plugins=[mock_plugin],
        prompt_fields=["text", "id"],
    )

    result = runner.run(sample_df)

    # Should process 2 rows
    assert len(result["results"]) == 2
    assert len(mock_plugin.calls) == 2
    assert mock_sink.written_payload is not None

    # Each result should have the processed field from the plugin
    for record in result["results"]:
        assert record["row"]["processed"] is True


def test_runner_skips_checkpointed_rows(sample_df, mock_sink, tmp_path):
    """SDARunner skips rows already in checkpoint."""
    checkpoint_path = tmp_path / "checkpoint.jsonl"

    # Pre-create checkpoint with id "1"
    checkpoint_path.write_text("1\n")

    mock_plugin = MockTransformPlugin()

    runner = SDARunner(
        sinks=[mock_sink],
        transform_plugins=[mock_plugin],
        prompt_fields=["text", "id"],
        checkpoint_config={"path": str(checkpoint_path), "field": "id"},
    )

    result = runner.run(sample_df)

    # Should only process row "2"
    assert len(result["results"]) == 1
    assert len(mock_plugin.calls) == 1


def test_runner_applies_transform_plugins(sample_df, mock_sink):
    """SDARunner applies transform plugins to each row."""
    mock_plugin = MockTransformPlugin()

    runner = SDARunner(
        sinks=[mock_sink],
        transform_plugins=[mock_plugin],
        prompt_fields=["text"],
    )

    result = runner.run(sample_df)

    # Transform plugin should be called for each row
    assert len(mock_plugin.calls) == 2
    # Results should include the transformed field
    assert result["results"][0]["row"]["processed"] is True


def test_runner_propagates_security_level(sample_df, mock_sink):
    """Security level is propagated to records and sink metadata."""
    sample_df.attrs["security_level"] = "secret"

    runner = SDARunner(
        sinks=[mock_sink],
        transform_plugins=[],
        prompt_fields=["text"],
        security_level="official-sensitive",  # explicit config wins over dataframe attr
    )

    result = runner.run(sample_df)

    # Records carry security level
    assert all(record.get("security_level") == "official-sensitive" for record in result["results"])
    # Payload metadata forwarded to sink includes security level
    assert mock_sink.written_metadata["security_level"] == "official-sensitive"


def test_runner_chains_multiple_plugins(sample_df, mock_sink):
    """SDARunner chains multiple transform plugins in sequence."""
    plugin1 = MockTransformPlugin(field_name="step1", field_value="done")
    plugin2 = MockTransformPlugin(field_name="step2", field_value="done")

    runner = SDARunner(
        sinks=[mock_sink],
        transform_plugins=[plugin1, plugin2],
        prompt_fields=["text"],
    )

    result = runner.run(sample_df)

    # Both plugins should have been called
    assert len(plugin1.calls) == 2
    assert len(plugin2.calls) == 2

    # Results should have both fields
    for record in result["results"]:
        assert record["row"]["step1"] == "done"
        assert record["row"]["step2"] == "done"


def test_runner_context_shared_between_plugins(sample_df, mock_sink):
    """Context dict is shared between plugins for inter-plugin communication."""

    class ContextWriterPlugin:
        name = "context_writer"

        def transform(self, row: dict, context: dict) -> dict:
            context["from_writer"] = row.get("id")
            return row

    class ContextReaderPlugin:
        name = "context_reader"
        read_values: list[Any] = []

        def transform(self, row: dict, context: dict) -> dict:
            self.read_values.append(context.get("from_writer"))
            row["context_value"] = context.get("from_writer")
            return row

    writer = ContextWriterPlugin()
    reader = ContextReaderPlugin()
    reader.read_values = []

    runner = SDARunner(
        sinks=[mock_sink],
        transform_plugins=[writer, reader],
        prompt_fields=["text", "id"],
    )

    runner.run(sample_df)

    # Reader should have seen values written by writer
    assert reader.read_values == ["1", "2"]


def test_runner_with_no_plugins(sample_df, mock_sink):
    """SDARunner works with no transform plugins."""
    runner = SDARunner(
        sinks=[mock_sink],
        transform_plugins=[],
        prompt_fields=["text", "id"],
    )

    result = runner.run(sample_df)

    # Should still process rows (pass-through)
    assert len(result["results"]) == 2
    assert mock_sink.written_payload is not None
