"""Integration tests for Docker operator workflow.

These tests verify the end-to-end workflow without actually running Docker.
They test the CLI with --secrets flag and config variable substitution.
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from elspeth import cli


@pytest.fixture
def workflow_setup(tmp_path):
    """Set up a complete workflow directory."""
    # Create input data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_file = data_dir / "input.csv"
    pd.DataFrame([
        {"id": 1, "text": "Hello world"},
        {"id": 2, "text": "Goodbye world"},
    ]).to_csv(input_file, index=False)

    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create config with variable placeholders
    config_file = tmp_path / "config.yaml"
    config = {
        "default": {
            "datasource": {
                "plugin": "local_csv",
                "options": {"path": "${INPUT_PATH}"},
            },
            "llm": {"plugin": "mock", "options": {}},
            "prompts": {"system": "Mock", "user": "Process: {text}"},
            "prompt_fields": ["text"],
            "sinks": [{
                "plugin": "csv",
                "options": {"path": "${OUTPUT_PATH}", "overwrite": True},
            }],
        }
    }
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    # Create secrets file
    secrets_file = tmp_path / "secrets.yaml"
    secrets = {
        "INPUT_PATH": str(input_file),
        "OUTPUT_PATH": str(output_dir / "results.csv"),
    }
    secrets_file.write_text(yaml.safe_dump(secrets), encoding="utf-8")

    return {
        "tmp_path": tmp_path,
        "config_file": config_file,
        "secrets_file": secrets_file,
        "input_file": input_file,
        "output_file": output_dir / "results.csv",
    }


def test_full_workflow_with_secrets(workflow_setup):
    """Test complete workflow: config + secrets -> output."""
    setup = workflow_setup

    cli.main([
        "--settings", str(setup["config_file"]),
        "--secrets", str(setup["secrets_file"]),
        "--log-level", "ERROR",
    ])

    assert setup["output_file"].exists()
    result_df = pd.read_csv(setup["output_file"])
    assert len(result_df) == 2
    assert "text" in result_df.columns


def test_workflow_fails_gracefully_without_secrets(workflow_setup, monkeypatch):
    """Test that missing secrets produces clear error."""
    setup = workflow_setup
    monkeypatch.chdir(setup["tmp_path"])

    # Remove secrets file
    setup["secrets_file"].unlink()

    # Create empty secrets
    setup["secrets_file"].write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli.main([
            "--settings", str(setup["config_file"]),
            "--secrets", str(setup["secrets_file"]),
        ])

    assert exc_info.value.code == 1

    # Diagnostics should be written
    diagnostics_dir = setup["tmp_path"] / "diagnostics"
    assert diagnostics_dir.exists()
