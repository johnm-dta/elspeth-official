

class MockAggregationPlugin:
    """Mock plugin that uses input_key/output_key."""
    def __init__(self, config):
        self.config = config
        self.name = config.get('name', 'mock')
        self.input_schema = {"type": "array"}
        self.output_schema = {"type": "object"}

    def aggregate(self, results, aggregates):
        """Aggregate method that receives aggregates dict."""
        # If has input_key, read from aggregates
        if 'input_key' in self.config:
            input_key = self.config['input_key']
            if input_key not in aggregates:
                raise KeyError(f"Missing input_key: {input_key}")
            data = aggregates[input_key]
            return {"processed": data}

        # Otherwise, process results normally
        return {"count": len(results)}


def test_result_aggregator_passes_aggregates_dict():
    """ResultAggregator should pass growing aggregates dict to plugins."""
    from elspeth.core.sda.result_aggregator import ResultAggregator

    # First plugin outputs to "data1"
    plugin1 = MockAggregationPlugin({"name": "plugin1", "output_key": "data1"})

    # Second plugin reads from "data1"
    plugin2 = MockAggregationPlugin({"name": "plugin2", "input_key": "data1"})

    aggregator = ResultAggregator(aggregation_plugins=[plugin1, plugin2])

    # Add some results
    aggregator.add_result({"score": 0.75}, row_index=0)
    aggregator.add_result({"score": 0.82}, row_index=1)

    # Build payload
    payload = aggregator.build_payload(
        security_level=None,
        early_stop_reason=None
    )

    # Check aggregates
    assert "aggregates" in payload
    assert "plugin1" in payload["aggregates"] or "data1" in payload["aggregates"]
    assert "plugin2" in payload["aggregates"]


def test_result_aggregator_stores_at_output_key():
    """ResultAggregator should store plugin output at output_key if specified."""
    from elspeth.core.sda.result_aggregator import ResultAggregator

    plugin = MockAggregationPlugin({"name": "test_plugin", "output_key": "custom_key"})

    aggregator = ResultAggregator(aggregation_plugins=[plugin])
    aggregator.add_result({"score": 0.75}, row_index=0)

    payload = aggregator.build_payload(
        security_level=None,
        early_stop_reason=None
    )

    # Should be stored at custom_key
    assert "custom_key" in payload["aggregates"]
    assert payload["aggregates"]["custom_key"] == {"count": 1}
