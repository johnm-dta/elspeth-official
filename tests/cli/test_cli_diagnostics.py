"""Tests for CLI diagnostics on error."""

from __future__ import annotations

import pytest
import yaml

from elspeth import cli


def test_diagnostics_written_on_config_error(tmp_path, monkeypatch):
    """Test that diagnostics are written when config fails."""
    # Change to tmp_path so diagnostics are written there
    monkeypatch.chdir(tmp_path)

    settings_path = tmp_path / "settings.yaml"
    # Invalid config - missing datasource
    settings = {
        "default": {
            "sinks": [],
        }
    }
    settings_path.write_text(yaml.safe_dump(settings), encoding="utf-8")

    with pytest.raises(SystemExit):
        cli.main(["--settings", str(settings_path)])

    diagnostics_dir = tmp_path / "diagnostics"
    assert diagnostics_dir.exists(), "Diagnostics directory should be created"
    assert (diagnostics_dir / "error-summary.txt").exists()


def test_diagnostics_written_on_secrets_error(tmp_path, monkeypatch):
    """Test diagnostics written when secrets missing."""
    monkeypatch.chdir(tmp_path)

    settings_path = tmp_path / "settings.yaml"
    settings = {
        "default": {
            "datasource": {
                "plugin": "local_csv",
                "options": {"path": "${MISSING_PATH}"},
            },
            "llm": {
                "plugin": "mock",
                "options": {},
            },
            "prompts": {
                "system": "Test",
                "user": "Test",
            },
            "sinks": [
                {
                    "plugin": "csv",
                    "options": {"path": "output.csv", "overwrite": True},
                }
            ],
        }
    }
    settings_path.write_text(yaml.safe_dump(settings), encoding="utf-8")

    secrets_path = tmp_path / "secrets.yaml"
    secrets_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        cli.main([
            "--settings", str(settings_path),
            "--secrets", str(secrets_path),
        ])

    diagnostics_dir = tmp_path / "diagnostics"
    assert diagnostics_dir.exists()

    error_summary = (diagnostics_dir / "error-summary.txt").read_text()
    assert "MISSING_PATH" in error_summary
