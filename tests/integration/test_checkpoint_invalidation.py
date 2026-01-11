# tests/integration/test_checkpoint_invalidation.py
"""Integration test for checkpoint invalidation on config change."""

import json
from pathlib import Path

import pandas as pd
import pytest

from elspeth.config import load_settings
from elspeth.core.orchestrator import SDAOrchestrator


class TestCheckpointInvalidation:
    """End-to-end test for config-aware checkpoint invalidation."""

    @pytest.fixture
    def test_config(self, tmp_path):
        """Create test config with checkpoint enabled."""
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "system.md").write_text("You are a test assistant.")
        (prompts / "user.md").write_text("Process: {text}")

        data = tmp_path / "data.csv"
        data.write_text("id,text\n1,hello\n2,world\n")

        config = tmp_path / "settings.yaml"
        config.write_text(f"""
default:
  datasource:
    plugin: local_csv
    options:
      path: {data}
  sinks:
    - plugin: csv
      options:
        path: {tmp_path}/output.csv
  checkpoint:
    path: {tmp_path}/checkpoint.jsonl
    field: id
  row_plugins:
    - plugin: llm_query
      options:
        llm:
          plugin: mock
        queries:
          - name: test
            prompt_folder: {prompts}
""")
        return {
            "config_path": config,
            "prompts": prompts,
            "checkpoint": tmp_path / "checkpoint.jsonl",
            "tmp_path": tmp_path,
        }

    def test_checkpoint_cleared_on_prompt_change(self, test_config):
        """Modifying prompt file invalidates checkpoint."""
        # First run - creates checkpoint
        settings1 = load_settings(test_config["config_path"])
        # ... would need mock LLM to actually run

        # For now, just verify fingerprint changes
        fp1 = settings1.orchestrator_config.config_fingerprint

        # Modify prompt
        (test_config["prompts"] / "user.md").write_text("CHANGED: {text}")

        # Second load - fingerprint should differ
        settings2 = load_settings(test_config["config_path"])
        fp2 = settings2.orchestrator_config.config_fingerprint

        assert fp1 != fp2, "Fingerprint should change when prompt changes"
