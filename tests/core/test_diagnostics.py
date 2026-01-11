"""Tests for diagnostics writer."""

from __future__ import annotations

from elspeth.core.diagnostics import DiagnosticsWriter


class TestDiagnosticsWriter:
    """Tests for DiagnosticsWriter."""

    def test_writes_error_summary(self, tmp_path):
        """Test error summary is written."""
        writer = DiagnosticsWriter(tmp_path / "diagnostics")
        writer.write_error(
            error_type="ConfigurationError",
            message="Missing required field: datasource",
            context={"file": "config.yaml"},
        )

        summary_path = tmp_path / "diagnostics" / "error-summary.txt"
        assert summary_path.exists()
        content = summary_path.read_text()
        assert "ConfigurationError" in content
        assert "Missing required field" in content

    def test_writes_redacted_config(self, tmp_path):
        """Test config is written with secrets redacted."""
        writer = DiagnosticsWriter(tmp_path / "diagnostics")
        config = {
            "datasource": {"path": "/data/input.csv"},
            "llm": {"api_key": "sk-secret123"},
            "storage": {"connection_string": "DefaultEndpointsProtocol=https;AccountKey=secret"},
        }
        writer.write_config(config)

        config_path = tmp_path / "diagnostics" / "config-resolved.yaml"
        assert config_path.exists()
        content = config_path.read_text()
        assert "sk-secret123" not in content
        assert "AccountKey=secret" not in content
        assert "[REDACTED]" in content

    def test_writes_stack_trace(self, tmp_path):
        """Test stack trace is captured."""
        writer = DiagnosticsWriter(tmp_path / "diagnostics")

        try:
            raise ValueError("Test error for diagnostics")
        except ValueError:
            writer.write_stack_trace()

        trace_path = tmp_path / "diagnostics" / "stack-trace.txt"
        assert trace_path.exists()
        content = trace_path.read_text()
        assert "ValueError" in content
        assert "Test error for diagnostics" in content

    def test_creates_directory_if_missing(self, tmp_path):
        """Test diagnostics directory is created."""
        diag_path = tmp_path / "nested" / "diagnostics"
        writer = DiagnosticsWriter(diag_path)
        writer.write_error("Error", "message", {})

        assert diag_path.exists()
        assert diag_path.is_dir()
