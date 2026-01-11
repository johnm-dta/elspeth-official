"""End-to-end integration test for landscape data collection."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from elspeth.core.landscape import RunLandscape, get_current_landscape


class RecordingLLM:
    """Mock LLM that records calls and returns predictable responses."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.call_count = 0

    def generate(
        self, system_prompt: str, user_prompt: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.call_count += 1
        self.calls.append({
            "system": system_prompt,
            "user": user_prompt,
            "metadata": metadata or {},
        })
        return {
            "content": f"Response {self.call_count}: processed",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


class TestLandscapeEndToEnd:
    """End-to-end tests verifying landscape collects all expected data."""

    def test_landscape_with_datasource_integration(self):
        """Datasource saves input data to landscape."""
        from elspeth.plugins.datasources.csv_local import CSVDataSource

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input CSV
            input_csv = Path(tmpdir) / "source.csv"
            input_csv.write_text("id,name,value\n1,alice,100\n2,bob,200\n")

            landscape_dir = Path(tmpdir) / "landscape"

            with RunLandscape(base_path=landscape_dir, persist=True) as landscape:
                # Load data through datasource
                source = CSVDataSource(path=str(input_csv), name="test_input")
                df = source.load()

                assert len(df) == 2

                # Verify data saved to landscape
                saved_path = landscape_dir / "inputs" / "test_input" / "source_data.csv"
                assert saved_path.exists(), "Datasource should save to landscape"

                saved_df = pd.read_csv(saved_path)
                assert len(saved_df) == 2
                assert list(saved_df.columns) == ["id", "name", "value"]

                # Check manifest
                manifest = landscape.get_manifest()
                input_artifacts = [a for a in manifest["artifacts"] if a["category"] == "inputs"]
                assert len(input_artifacts) == 1
                assert input_artifacts[0]["metadata"]["rows"] == 2

            # Cleanup
            shutil.rmtree(landscape_dir)

    def test_landscape_with_archive_bundle(self):
        """ArchiveBundleSink includes landscape in archive."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir) / "archives"
            archive_dir.mkdir()

            with RunLandscape() as landscape:
                # Add artifacts to landscape
                input_path = landscape.get_path("inputs", "data_source", "input.csv")
                input_path.write_text("a,b\n1,2\n")
                landscape.register_artifact("inputs", "data_source", input_path, {"rows": 1})

                # Log an LLM call
                landscape.log_llm_call(
                    "call_001",
                    {"system_prompt": "Be helpful", "user_prompt": "Hello"},
                    {"content": "Hi there!"},
                    {"row_id": 1},
                )

                # Save config
                landscape.save_config([], {"test": "config"})

                # Create archive
                import os
                os.environ.setdefault("DMP_ARCHIVE_SIGNING_KEY", "test-key-12345")

                sink = ArchiveBundleSink(
                    base_path=archive_dir,
                    archive_name="e2e_test",
                    timestamped=False,
                    project_root=Path(tmpdir),
                )

                sink.write({"results": [{"id": 1}]}, metadata={})

                # Verify archive contains landscape
                archive_path = sink._last_archive
                assert archive_path is not None

                with ZipFile(archive_path) as zf:
                    names = zf.namelist()

                    # Check landscape structure in archive
                    assert "landscape/manifest.json" in names, "Should have landscape manifest"

                    # Check inputs
                    input_entries = [n for n in names if "landscape/inputs/" in n]
                    assert len(input_entries) > 0, "Should have landscape inputs"

                    # Check LLM calls
                    llm_entries = [n for n in names if "landscape/intermediate/llm_calls/" in n]
                    assert len(llm_entries) > 0, "Should have LLM calls"

                    # Check config
                    config_entries = [n for n in names if "landscape/config/" in n]
                    assert len(config_entries) > 0, "Should have config"

                    # Verify manifest content
                    manifest_data = json.loads(zf.read("landscape/manifest.json"))
                    assert "artifacts" in manifest_data
                    assert len(manifest_data["artifacts"]) >= 1

    def test_landscape_cleanup_on_error(self):
        """Landscape cleans up properly even when errors occur."""
        landscape_root = None

        try:
            with RunLandscape() as landscape:
                landscape_root = landscape.root
                assert landscape_root.exists()

                # Simulate error
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Landscape should be cleaned up
        assert landscape_root is not None
        assert not landscape_root.exists(), "Landscape should be cleaned up on error"

    def test_landscape_persist_preserves_data(self):
        """persist=True keeps landscape after context exit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            landscape_dir = Path(tmpdir) / "persistent"

            with RunLandscape(base_path=landscape_dir, persist=True) as landscape:
                # Add some data
                test_path = landscape.get_path("intermediate", "test", "data.txt")
                test_path.write_text("test data")
                landscape.register_artifact("intermediate", "test", test_path, {})
                landscape.write_manifest()

            # After context exit, data should still exist
            assert landscape_dir.exists()
            assert (landscape_dir / "intermediate" / "test" / "data.txt").exists()
            assert (landscape_dir / "manifest.json").exists()

            # Cleanup
            shutil.rmtree(landscape_dir)

    def test_no_landscape_graceful_handling(self):
        """Components work correctly when no landscape is active."""
        from elspeth.core.sda.llm_executor import LLMExecutor
        from elspeth.plugins.datasources.csv_local import CSVDataSource

        with tempfile.TemporaryDirectory() as tmpdir:
            # Datasource without landscape
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("x,y\n1,2\n")

            source = CSVDataSource(path=str(csv_path))
            df = source.load()
            assert len(df) == 1

            # LLM executor without landscape
            llm = RecordingLLM()
            executor = LLMExecutor(
                llm_client=llm,
                middlewares=[],
                retry_config=None,
                rate_limiter=None,
                cost_tracker=None,
            )

            result = executor.execute(
                system_prompt="Test",
                user_prompt="Hello",
                metadata={},
            )
            assert "content" in result

            # No landscape should be active
            assert get_current_landscape() is None
