"""Tests for ResultAggregator runtime validation."""

from typing import Any, ClassVar

import pytest

from elspeth.core.sda.result_aggregator import ResultAggregator
from elspeth.core.validation import ConfigurationError


class MockAggregationPlugin:
    """Mock aggregation plugin with schemas."""
    def __init__(self, input_schema, output_schema=None):
        self.name = "mock_aggregator"
        self.config = {}
        self.input_schema = input_schema
        self.output_schema = output_schema or {
            "type": "object",
            "properties": {
                "total": {"type": "number"}
            }
        }

    def aggregate(self, results, aggregates):
        # Return aggregated metrics matching output_schema
        return {"total": len(results)}


class BuggyAggregator:
    """Aggregator that violates its schema."""
    name = "buggy"
    config: ClassVar[dict[str, Any]] = {}
    input_schema: ClassVar[dict[str, Any]] = {"type": "array"}
    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["mean"],
        "properties": {
            "mean": {"type": "number"}
        }
    }

    def aggregate(self, results, aggregates):
        # Bug: returns string instead of number
        return {"mean": "not_a_number"}


def test_result_aggregator_validates_plugin_output():
    """ResultAggregator should validate aggregation plugin output."""
    plugin = BuggyAggregator()
    aggregator = ResultAggregator(aggregation_plugins=[plugin])

    # Add some results
    aggregator.add_result({"row": {"score": 0.75}}, row_index=0)
    aggregator.add_result({"row": {"score": 0.82}}, row_index=1)

    # Build payload should raise ConfigurationError due to buggy output
    with pytest.raises(ConfigurationError) as exc_info:
        aggregator.build_payload()

    assert "mean" in str(exc_info.value).lower()
    assert "number" in str(exc_info.value).lower()


def test_result_aggregator_accepts_valid_output():
    """ResultAggregator should accept valid aggregation plugin output."""
    plugin = MockAggregationPlugin(
        input_schema={"type": "array"}
    )
    aggregator = ResultAggregator(aggregation_plugins=[plugin])

    # Add some results
    aggregator.add_result({"row": {"score": 0.75}}, row_index=0)
    aggregator.add_result({"row": {"score": 0.82}}, row_index=1)

    # Build payload should succeed
    payload = aggregator.build_payload()

    assert "aggregates" in payload
    assert "mock_aggregator" in payload["aggregates"]
    assert payload["aggregates"]["mock_aggregator"]["total"] == 2


def test_result_aggregator_skips_validation_without_output_schema():
    """ResultAggregator should skip validation if plugin has no schemas."""
    class SimpleAggregator:
        """Aggregator without any schemas."""
        name = "simple"
        config: ClassVar[dict[str, Any]] = {}
        # No input_schema or output_schema - validation should be skipped

        def aggregate(self, results, aggregates):
            return {"count": len(results)}

    plugin = SimpleAggregator()
    aggregator = ResultAggregator(aggregation_plugins=[plugin])

    aggregator.add_result({"row": {}}, row_index=0)

    # Should work without errors (validation skipped)
    payload = aggregator.build_payload()
    assert "aggregates" in payload
    assert payload["aggregates"]["simple"]["count"] == 1
