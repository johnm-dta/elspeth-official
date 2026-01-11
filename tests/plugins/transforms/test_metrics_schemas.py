"""Tests for metrics plugin schemas (Task 1: row-mode plugins)."""

import pytest

from elspeth.core.validation import validate_schema
from elspeth.plugins.transforms.metrics import (
    ScoreAssumptionsBaselinePlugin,
    ScoreBayesianBaselinePlugin,
    ScoreCliffsDeltaPlugin,
    ScoreDeltaBaselinePlugin,
    ScoreExtractorPlugin,
    ScorePracticalBaselinePlugin,
    ScoreSignificanceBaselinePlugin,
)


def test_score_extractor_has_schemas():
    """ScoreExtractorPlugin should have input and output schemas."""
    assert hasattr(ScoreExtractorPlugin, 'config_schema')
    assert hasattr(ScoreExtractorPlugin, 'input_schema')
    assert hasattr(ScoreExtractorPlugin, 'output_schema')

    # Input: row object with criteria responses
    assert ScoreExtractorPlugin.input_schema['type'] == 'object'

    # Output: row object with extracted scores
    assert ScoreExtractorPlugin.output_schema['type'] == 'object'


def test_score_extractor_input_schema_validation():
    """ScoreExtractorPlugin input_schema should accept valid row data."""
    valid_row = {
        "id": 1,
        "response": {"score": 0.75}
    }

    errors = list(validate_schema(valid_row, ScoreExtractorPlugin.input_schema))
    assert len(errors) == 0


def test_score_extractor_output_schema_validation():
    """ScoreExtractorPlugin output_schema should accept transformed row."""
    # Row with extracted score
    output_row = {
        "id": 1,
        "response": {"score": 0.75},
        "extracted_score": 0.75
    }

    errors = list(validate_schema(output_row, ScoreExtractorPlugin.output_schema))
    assert len(errors) == 0


@pytest.mark.parametrize("plugin_class", [
    ScoreDeltaBaselinePlugin,
    ScoreCliffsDeltaPlugin,
    ScoreAssumptionsBaselinePlugin,
    ScorePracticalBaselinePlugin,
    ScoreSignificanceBaselinePlugin,
    ScoreBayesianBaselinePlugin,
])
def test_comparison_plugin_has_schemas(plugin_class):
    """All comparison plugins should have required schemas."""
    assert hasattr(plugin_class, 'config_schema')
    assert hasattr(plugin_class, 'input_schema')
    assert hasattr(plugin_class, 'output_schema')

    # Input: row object
    assert plugin_class.input_schema['type'] == 'object'

    # Output: row object (adds comparison metrics)
    assert plugin_class.output_schema['type'] == 'object'
