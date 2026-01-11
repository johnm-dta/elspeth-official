"""Tests for compile-time pipeline validation."""

from elspeth.core.pipeline_validator import PipelineValidator


class MockRowPlugin:
    """Mock row plugin for testing."""
    def __init__(self, input_schema, output_schema=None):
        self.input_schema = input_schema
        self.output_schema = output_schema or input_schema


class MockAggregationPlugin:
    """Mock aggregation plugin for testing."""
    def __init__(self, input_schema, output_schema=None, config=None):
        self.input_schema = input_schema
        self.output_schema = output_schema or input_schema
        self.config = config or {}


def test_pipeline_validator_validates_compatible_schemas():
    """Pipeline validator should accept compatible row plugin chain."""
    validator = PipelineValidator()

    plugins = [
        MockRowPlugin(
            input_schema={"type": "object", "properties": {"score": {"type": "number"}}},
            output_schema={"type": "object", "properties": {"score": {"type": "number"}, "delta": {"type": "number"}}}
        ),
        MockRowPlugin(
            input_schema={"type": "object", "properties": {"score": {"type": "number"}, "delta": {"type": "number"}}},
            output_schema={"type": "object", "properties": {"score": {"type": "number"}, "delta": {"type": "number"}, "rank": {"type": "integer"}}}
        )
    ]

    errors = validator.validate_row_plugin_chain(plugins)
    assert len(errors) == 0


def test_pipeline_validator_detects_type_mismatch():
    """Pipeline validator should detect execution mode mismatches."""
    validator = PipelineValidator()

    # Plugin A outputs object, Plugin B expects collection
    plugin_a = MockRowPlugin(
        input_schema={"type": "object"},
        output_schema={"type": "object"}
    )

    plugin_b = MockAggregationPlugin(
        input_schema={"type": "collection", "item_schema": {"type": "object"}}
    )

    errors = validator.validate_execution_mode_transition(plugin_a, plugin_b)
    assert len(errors) > 0
    assert "execution mode" in errors[0].message.lower() or "type mismatch" in errors[0].message.lower()


def test_pipeline_validator_validates_dependency_graph():
    """Pipeline validator should check input_key/output_key dependencies."""
    validator = PipelineValidator()

    plugins = [
        MockAggregationPlugin(
            input_schema={"type": "array"},
            output_schema={"type": "collection"},
            config={"output_key": "collected_scores"}
        ),
        MockAggregationPlugin(
            input_schema={"type": "collection"},
            output_schema={"type": "object"},
            config={"input_key": "missing_key"}  # This key doesn't exist!
        )
    ]

    errors = validator.validate_dependency_graph(plugins)
    assert len(errors) > 0
    assert "missing_key" in errors[0].message


def test_pipeline_validator_checks_required_fields():
    """Pipeline validator should check if required fields are present."""
    validator = PipelineValidator()

    plugin_a = MockRowPlugin(
        input_schema={"type": "object"},
        output_schema={
            "type": "object",
            "properties": {
                "score": {"type": "number"}
            }
        }
    )

    plugin_b = MockRowPlugin(
        input_schema={
            "type": "object",
            "required": ["score", "delta"],  # Requires delta!
            "properties": {
                "score": {"type": "number"},
                "delta": {"type": "number"}
            }
        }
    )

    errors = validator.validate_field_requirements(plugin_a, plugin_b)
    assert len(errors) > 0
    assert "delta" in errors[0].message
