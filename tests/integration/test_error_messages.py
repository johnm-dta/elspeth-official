"""Tests for error message quality and clarity."""

import pytest

from elspeth.core.validation import validate_schema


class TestCompileTimeErrorMessages:
    """Test compile-time validation error messages are clear."""

    def test_missing_required_field_message(self):
        """Missing required field error is clear."""
        schema = {
            "type": "object",
            "required": ["score", "name"],
            "properties": {
                "score": {"type": "number"},
                "name": {"type": "string"},
            }
        }

        data = {"score": 0.8}  # Missing "name"
        errors = list(validate_schema(data, schema))

        assert len(errors) == 1
        error_msg = str(errors[0])

        # Should clearly identify what's missing
        assert "name" in error_msg.lower() or "required" in error_msg.lower()

    def test_type_mismatch_message(self):
        """Type mismatch error identifies expected vs actual."""
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
            }
        }

        data = {"score": "not a number"}
        errors = list(validate_schema(data, schema))

        assert len(errors) >= 1
        error_msg = str(errors[0])

        # Should identify field and type issue
        assert "score" in error_msg.lower() or "number" in error_msg.lower()

    def test_collection_type_error_message(self):
        """Collection validation error is clear."""
        schema = {
            "type": "collection",
            "item_schema": {
                "type": "object",
                "properties": {"score": {"type": "number"}}
            }
        }

        # Not a collection (dict of arrays)
        data = [{"score": 0.8}]  # Array, not collection
        errors = list(validate_schema(data, schema))

        # Should have error about collection type
        assert len(errors) >= 1


class TestRuntimeErrorMessages:
    """Test runtime error messages are actionable."""

    def test_analyzer_missing_collection_suggests_fix(self):
        """Analyzer error suggests using FieldCollector."""
        from elspeth.plugins.transforms.metrics import ScoreStatsAnalyzer

        analyzer = ScoreStatsAnalyzer(
            input_key="missing_data",
            source_field="score"
        )

        with pytest.raises(KeyError) as exc_info:
            analyzer.aggregate([], aggregates={})

        error_msg = str(exc_info.value)

        # Should suggest the solution
        assert "FieldCollector" in error_msg
        assert "output_key" in error_msg

    def test_analyzer_missing_field_lists_available(self):
        """Missing field error lists available fields."""
        from elspeth.plugins.transforms.field_collector import FieldCollector
        from elspeth.plugins.transforms.metrics import ScoreStatsAnalyzer

        # Create collection with different field
        collector = FieldCollector({"output_key": "data"})
        rows = [{"rating": 4.5}, {"rating": 4.8}]
        collection = collector.aggregate(rows, {})

        analyzer = ScoreStatsAnalyzer(
            input_key="data",
            source_field="score"  # Wrong field name
        )

        with pytest.raises(ValueError) as exc_info:
            analyzer.aggregate([], aggregates={"data": collection})

        error_msg = str(exc_info.value)

        # Should list available fields
        assert "Available fields" in error_msg or "rating" in error_msg.lower()

    def test_expander_missing_input_key_error(self):
        """FieldExpander gives clear error for missing input_key."""
        from elspeth.plugins.transforms.field_expander import FieldExpander

        expander = FieldExpander({"input_key": "nonexistent"})

        with pytest.raises(KeyError) as exc_info:
            expander.aggregate([], aggregates={"other": {}})

        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg


class TestConfigurationErrorMessages:
    """Test configuration validation error messages."""

    def test_missing_required_option_message(self):
        """Missing required config option error is clear."""
        from elspeth.core.validation import validate_schema

        config_schema = {
            "type": "object",
            "required": ["input_key", "source_field"],
            "properties": {
                "input_key": {"type": "string"},
                "source_field": {"type": "string"},
            }
        }

        # Missing input_key
        config = {"source_field": "score"}
        errors = list(validate_schema(config, config_schema))

        assert len(errors) >= 1
        error_msg = " ".join(str(e) for e in errors)
        assert "input_key" in error_msg.lower() or "required" in error_msg.lower()

    def test_invalid_option_type_message(self):
        """Invalid config option type error is clear."""
        from elspeth.core.validation import validate_schema

        config_schema = {
            "type": "object",
            "properties": {
                "min_samples": {"type": "integer"},
            }
        }

        config = {"min_samples": "not a number"}
        errors = list(validate_schema(config, config_schema))

        assert len(errors) >= 1
        error_msg = str(errors[0])
        # Should mention the field or type
        assert "min_samples" in error_msg.lower() or "integer" in error_msg.lower()
