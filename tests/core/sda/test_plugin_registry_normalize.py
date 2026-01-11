"""Tests for plugin registry normalization helpers."""

import pytest

from elspeth.core.sda.plugin_registry import normalize_halt_condition_definitions
from elspeth.core.validation import ConfigurationError


def test_normalize_halt_condition_definitions_from_mapping():
    normalized = normalize_halt_condition_definitions({"name": "threshold", "options": {"metric": "score"}})
    assert normalized == [{"name": "threshold", "options": {"metric": "score"}}]


def test_normalize_halt_condition_definitions_inline_options():
    normalized = normalize_halt_condition_definitions({"name": "threshold", "metric": "score", "threshold": 0.5})
    assert normalized == [{"name": "threshold", "options": {"metric": "score", "threshold": 0.5}}]


def test_normalize_halt_condition_definitions_sequence():
    normalized = normalize_halt_condition_definitions([{"plugin": "threshold", "options": {"metric": "score"}}])
    assert normalized == [{"name": "threshold", "options": {"metric": "score"}}]


def test_normalize_halt_condition_definitions_invalid_type_raises():
    with pytest.raises(ConfigurationError):
        normalize_halt_condition_definitions("not a mapping")
