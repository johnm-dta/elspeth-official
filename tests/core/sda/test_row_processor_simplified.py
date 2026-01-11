"""Tests for simplified RowProcessor (generic plugin executor)."""

from __future__ import annotations

from typing import Any

import pandas as pd


class MockPlugin:
    """Mock transform plugin for testing."""

    name = "mock"

    def __init__(self, transform_fn=None):
        self._transform_fn = transform_fn or (lambda row, ctx: row)

    def transform(self, row: dict, context: dict) -> dict:
        return self._transform_fn(row, context)


class TestRowProcessorSimplified:
    """Test simplified RowProcessor."""

    def test_process_row_runs_plugins_in_sequence(self):
        """RowProcessor runs plugins in sequence, passing context."""
        from elspeth.core.sda.row_processor import RowProcessor

        call_order = []

        def plugin1_transform(row, ctx):
            call_order.append("plugin1")
            ctx["from_plugin1"] = "value1"
            row["field1"] = "added_by_plugin1"
            return row

        def plugin2_transform(row, ctx):
            call_order.append("plugin2")
            # Can see plugin1's context
            assert ctx.get("from_plugin1") == "value1"
            ctx["from_plugin2"] = "value2"
            row["field2"] = "added_by_plugin2"
            return row

        plugins = [
            MockPlugin(plugin1_transform),
            MockPlugin(plugin2_transform),
        ]

        processor = RowProcessor(transform_plugins=plugins)

        row_series = pd.Series({"original": "data"})
        row_data = {"original": "data"}

        record, failure = processor.process_row(row_series, row_data, row_id="123")

        assert failure is None
        assert record is not None
        assert call_order == ["plugin1", "plugin2"]
        assert record["row"]["field1"] == "added_by_plugin1"
        assert record["row"]["field2"] == "added_by_plugin2"
        assert record["context"]["from_plugin1"] == "value1"
        assert record["context"]["from_plugin2"] == "value2"

    def test_process_row_with_empty_plugins(self):
        """RowProcessor returns row unchanged when no plugins configured."""
        from elspeth.core.sda.row_processor import RowProcessor

        processor = RowProcessor(transform_plugins=[])

        row_series = pd.Series({"original": "data"})
        row_data = {"original": "data"}

        record, failure = processor.process_row(row_series, row_data, row_id="123")

        assert failure is None
        assert record is not None
        assert record["row"] == {"original": "data"}
        assert record["context"] == {}

    def test_process_row_handles_plugin_error(self):
        """RowProcessor captures plugin errors as failures."""
        from elspeth.core.sda.row_processor import RowProcessor

        def failing_transform(row, ctx):
            raise ValueError("Plugin error")

        plugins = [MockPlugin(failing_transform)]
        processor = RowProcessor(transform_plugins=plugins)

        row_series = pd.Series({"original": "data"})
        row_data = {"original": "data"}

        record, failure = processor.process_row(row_series, row_data, row_id="123")

        assert record is None
        assert failure is not None
        assert "Plugin error" in failure["error"]
        assert failure["error_type"] == "ValueError"

    def test_process_row_adds_security_level(self):
        """RowProcessor adds security level to record when configured."""
        from elspeth.core.sda.row_processor import RowProcessor

        processor = RowProcessor(
            transform_plugins=[],
            security_level="OFFICIAL:SENSITIVE",
        )

        row_series = pd.Series({"data": "value"})
        row_data = {"data": "value"}

        record, failure = processor.process_row(row_series, row_data, row_id="123")

        assert failure is None
        assert record is not None
        assert record["security_level"] == "OFFICIAL:SENSITIVE"

    def test_process_row_validates_input_schema(self):
        """RowProcessor validates row against plugin input_schema."""
        from elspeth.core.sda.row_processor import RowProcessor

        class PluginWithSchema:
            name = "schema_plugin"
            input_schema = {
                "type": "object",
                "required": ["score"],
                "properties": {"score": {"type": "number"}},
            }

            def transform(self, row: dict, context: dict) -> dict:
                return row

        processor = RowProcessor(transform_plugins=[PluginWithSchema()])

        # Valid input
        row_series = pd.Series({"score": 0.75})
        row_data = {"score": 0.75}
        record, failure = processor.process_row(row_series, row_data, row_id="1")
        assert record is not None
        assert failure is None

        # Invalid input (missing required field)
        row_series = pd.Series({"name": "test"})
        row_data = {"name": "test"}
        record, failure = processor.process_row(row_series, row_data, row_id="2")
        assert record is None
        assert failure is not None
        assert "score" in str(failure["error"]).lower()

    def test_process_row_validates_output_schema(self):
        """RowProcessor validates plugin output against output_schema."""
        from elspeth.core.sda.row_processor import RowProcessor

        class BuggyPlugin:
            name = "buggy_plugin"
            input_schema: dict[str, Any] = {"type": "object"}
            output_schema: dict[str, Any] = {
                "type": "object",
                "required": ["result"],
                "properties": {"result": {"type": "number"}},
            }

            def transform(self, row: dict, context: dict) -> dict:
                # Bug: returns string instead of number
                return {"result": "not_a_number"}

        processor = RowProcessor(transform_plugins=[BuggyPlugin()])

        row_series = pd.Series({})
        row_data = {}
        record, failure = processor.process_row(row_series, row_data, row_id="1")

        assert record is None
        assert failure is not None
        assert "number" in str(failure["error"]).lower()
