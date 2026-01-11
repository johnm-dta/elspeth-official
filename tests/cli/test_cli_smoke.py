"""Smoke test for CLI with mock plugins."""

from __future__ import annotations

import pandas as pd
import yaml

from elspeth import cli


def test_cli_runs_with_mock_plugins(tmp_path, capsys):
    data_path = tmp_path / "data.csv"
    df = pd.DataFrame([{"text": "hello"}, {"text": "world"}])
    df.to_csv(data_path, index=False)

    output_path = tmp_path / "results.csv"

    settings_path = tmp_path / "settings.yaml"
    settings = {
        "default": {
            "datasource": {
                "plugin": "local_csv",
                "options": {"path": str(data_path)},
            },
            "llm": {
                "plugin": "mock",
                "options": {},
            },
            "prompts": {
                "system": "You are a mock system.",
                "user": "Say hi to {text}",
            },
            "prompt_fields": ["text"],
            "sinks": [
                {"plugin": "csv", "options": {"path": str(output_path), "overwrite": True}},
            ],
        }
    }
    settings_path.write_text(yaml.safe_dump(settings), encoding="utf-8")

    cli.main(
        [
            "--settings",
            str(settings_path),
            "--head",
            "0",  # avoid printing previews
            "--log-level",
            "ERROR",
        ]
    )

    captured = capsys.readouterr()
    assert output_path.exists()
    result_df = pd.read_csv(output_path)
    assert len(result_df) == 2
    # Ensure CLI did not emit error output
    assert captured.err == ""
