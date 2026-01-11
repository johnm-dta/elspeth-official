"""Tests for SDARunner fingerprint integration."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from elspeth.core.sda.runner import SDARunner


class TestSDARunnerFingerprint:
    """Tests for fingerprint-based checkpoint invalidation in SDARunner."""

    @pytest.fixture
    def mock_sink(self):
        """Create mock sink."""
        sink = MagicMock()
        sink.produces.return_value = []
        sink.consumes.return_value = []
        # Configure _dmp attributes to avoid security level issues
        sink._dmp_security_level = None
        sink._dmp_artifact_config = {}
        sink._dmp_plugin_name = "mock_sink"
        sink._dmp_sink_name = "mock_sink"
        return sink

    @pytest.fixture
    def simple_df(self):
        """Simple test DataFrame."""
        return pd.DataFrame({"id": ["row1", "row2"], "text": ["a", "b"]})

    def test_checkpoint_uses_fingerprint(self, tmp_path, mock_sink, simple_df):
        """SDARunner passes fingerprint to CheckpointManager."""
        checkpoint_path = tmp_path / "checkpoint.jsonl"

        runner = SDARunner(
            sinks=[mock_sink],
            checkpoint_config={
                "path": str(checkpoint_path),
                "field": "id",
            },
            config_fingerprint="test_fingerprint_123",
        )

        runner.run(simple_df)

        # Meta file should exist with fingerprint
        meta_path = checkpoint_path.with_suffix(".jsonl.meta")
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["fingerprint"] == "test_fingerprint_123"
