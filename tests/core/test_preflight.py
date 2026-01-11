"""Tests for preflight validation."""

from __future__ import annotations

from elspeth.core.preflight import PreflightValidator


class TestPreflightValidator:
    """Tests for PreflightValidator."""

    def test_passes_when_no_unresolved_variables(self):
        """Test validation passes with fully resolved config."""
        config = {"datasource": {"options": {"path": "/data/input.csv"}}}
        validator = PreflightValidator(config)
        result = validator.validate()
        assert result.success
        assert not result.errors

    def test_fails_with_unresolved_variables(self):
        """Test validation fails when ${VAR} placeholders remain."""
        config = {"datasource": {"options": {"path": "${DATA_PATH}"}}}
        validator = PreflightValidator(config)
        result = validator.validate()
        assert not result.success
        assert any("DATA_PATH" in err for err in result.errors)

    def test_reports_all_unresolved_variables(self):
        """Test all unresolved variables are reported."""
        config = {
            "datasource": {"connection": "${CONN_STRING}"},
            "llm": {"api_key": "${API_KEY}"},
        }
        validator = PreflightValidator(config)
        result = validator.validate()
        assert not result.success
        assert any("CONN_STRING" in err for err in result.errors)
        assert any("API_KEY" in err for err in result.errors)

    def test_result_has_human_readable_summary(self):
        """Test result provides operator-friendly summary."""
        config = {"key": "${MISSING}"}
        validator = PreflightValidator(config)
        result = validator.validate()
        summary = result.summary()
        assert "MISSING" in summary
        assert "secrets.yaml" in summary.lower() or "secret" in summary.lower()
