"""Tests for --secrets CLI flag."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from elspeth import cli


def test_cli_accepts_secrets_flag(tmp_path):
    """Test that CLI accepts --secrets argument."""
    data_path = tmp_path / "data.csv"
    df = pd.DataFrame([{"text": "hello"}])
    df.to_csv(data_path, index=False)

    output_path = tmp_path / "results.csv"

    settings_path = tmp_path / "settings.yaml"
    settings = {
        "default": {
            "datasource": {
                "plugin": "local_csv",
                "options": {"path": str(data_path)},
            },
            "llm": {"plugin": "mock", "options": {}},
            "prompts": {"system": "Mock", "user": "Hi {text}"},
            "prompt_fields": ["text"],
            "sinks": [{"plugin": "csv", "options": {"path": str(output_path), "overwrite": True}}],
        }
    }
    settings_path.write_text(yaml.safe_dump(settings), encoding="utf-8")

    secrets_path = tmp_path / "secrets.yaml"
    secrets_path.write_text(yaml.safe_dump({"test_key": "test_value"}), encoding="utf-8")

    # Should not raise - just verify flag is accepted
    cli.main([
        "--settings", str(settings_path),
        "--secrets", str(secrets_path),
        "--head", "0",
        "--log-level", "ERROR",
    ])

    assert output_path.exists()


def test_secrets_substitution_in_config(tmp_path):
    """Test that secrets are substituted into config values."""
    data_path = tmp_path / "data.csv"
    df = pd.DataFrame([{"text": "hello"}])
    df.to_csv(data_path, index=False)

    output_path = tmp_path / "results.csv"

    # Config uses ${DATA_PATH} variable
    settings_path = tmp_path / "settings.yaml"
    settings = {
        "default": {
            "datasource": {
                "plugin": "local_csv",
                "options": {"path": "${DATA_PATH}"},
            },
            "llm": {"plugin": "mock", "options": {}},
            "prompts": {"system": "Mock", "user": "Hi {text}"},
            "prompt_fields": ["text"],
            "sinks": [{"plugin": "csv", "options": {"path": str(output_path), "overwrite": True}}],
        }
    }
    settings_path.write_text(yaml.safe_dump(settings), encoding="utf-8")

    # Secrets provides the value
    secrets_path = tmp_path / "secrets.yaml"
    secrets_path.write_text(yaml.safe_dump({"DATA_PATH": str(data_path)}), encoding="utf-8")

    cli.main([
        "--settings", str(settings_path),
        "--secrets", str(secrets_path),
        "--head", "1",
        "--log-level", "ERROR",
    ])

    assert output_path.exists()
    result_df = pd.read_csv(output_path)
    assert len(result_df) == 1


def test_missing_secret_raises_clear_error(tmp_path, capsys):
    """Test that missing secrets produce clear error messages."""
    output_path = tmp_path / "results.csv"

    settings_path = tmp_path / "settings.yaml"
    settings = {
        "default": {
            "datasource": {
                "plugin": "local_csv",
                "options": {"path": "${MISSING_PATH}"},
            },
            "llm": {"plugin": "mock", "options": {}},
            "prompts": {"system": "Mock", "user": "Hi"},
            "sinks": [{"plugin": "csv", "options": {"path": str(output_path), "overwrite": True}}],
        }
    }
    settings_path.write_text(yaml.safe_dump(settings), encoding="utf-8")

    secrets_path = tmp_path / "secrets.yaml"
    secrets_path.write_text(yaml.safe_dump({}), encoding="utf-8")

    with pytest.raises(SystemExit):
        cli.main([
            "--settings", str(settings_path),
            "--secrets", str(secrets_path),
        ])
