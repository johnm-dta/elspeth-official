"""Tests for ResultAggregator."""

from elspeth.core.sda.result_aggregator import ResultAggregator


class MockAggregationPlugin:
    # No schemas - this mock is used to test various return types (dict, list, None)
    # Schema validation is tested separately in test_result_aggregator_validation.py

    def __init__(self, name: str, aggregated: dict):
        self.name = name
        self.config = {}
        self.aggregated = aggregated

    def aggregate(self, results: list, aggregates: dict) -> dict:
        return self.aggregated


def test_result_aggregator_collects_results():
    """ResultAggregator collects results and failures."""
    aggregator = ResultAggregator(aggregation_plugins=[])

    aggregator.add_result({"id": 1}, row_index=0)
    aggregator.add_result({"id": 2}, row_index=1)
    aggregator.add_failure({"error": "test"})

    payload = aggregator.build_payload(security_level="OFFICIAL-SENSITIVE")

    assert len(payload["results"]) == 2
    assert payload["results"][0]["id"] == 1
    assert payload["results"][1]["id"] == 2
    assert len(payload["failures"]) == 1
    assert payload["metadata"]["rows"] == 2


def test_result_aggregator_applies_aggregation_plugins():
    """ResultAggregator applies aggregation plugins."""
    plugin = MockAggregationPlugin("stats", {"mean": 5.0})
    aggregator = ResultAggregator(aggregation_plugins=[plugin])

    aggregator.add_result({"value": 3}, row_index=0)
    aggregator.add_result({"value": 7}, row_index=1)

    payload = aggregator.build_payload()

    assert "aggregates" in payload
    assert payload["aggregates"]["stats"]["mean"] == 5.0
    assert payload["metadata"]["aggregates"]["stats"]["mean"] == 5.0


def test_result_aggregator_preserves_empty_dict():
    """ResultAggregator preserves empty dict outputs (not discarded as falsy)."""
    # FieldCollector returns {} when there are no rows
    plugin = MockAggregationPlugin("field_collector", {})
    aggregator = ResultAggregator(aggregation_plugins=[plugin])

    # No results added
    payload = aggregator.build_payload()

    # Empty dict should be stored, not discarded
    assert "aggregates" in payload
    assert "field_collector" in payload["aggregates"]
    assert payload["aggregates"]["field_collector"] == {}


def test_result_aggregator_preserves_empty_list():
    """ResultAggregator preserves empty list outputs (not discarded as falsy)."""
    plugin = MockAggregationPlugin("expander", [])
    aggregator = ResultAggregator(aggregation_plugins=[plugin])

    aggregator.add_result({"value": 1}, row_index=0)
    payload = aggregator.build_payload()

    # Empty list should be stored, not discarded
    assert "aggregates" in payload
    assert "expander" in payload["aggregates"]
    assert payload["aggregates"]["expander"] == []


def test_result_aggregator_discards_none():
    """ResultAggregator discards None outputs (plugin explicitly returns None)."""
    plugin = MockAggregationPlugin("optional", None)
    aggregator = ResultAggregator(aggregation_plugins=[plugin])

    aggregator.add_result({"value": 1}, row_index=0)
    payload = aggregator.build_payload()

    # None should be discarded
    if "aggregates" in payload:
        assert "optional" not in payload["aggregates"]
