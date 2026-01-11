"""Tests for RunLandscape artifact management."""

import shutil
import tempfile
from pathlib import Path

import pytest

from elspeth.core.landscape import RunLandscape


class TestRunLandscapeCreation:
    """Test landscape directory creation."""

    def test_creates_temp_directory_when_no_path_given(self):
        """Landscape creates temp directory if base_path not provided."""
        landscape = RunLandscape()
        try:
            assert landscape.root.exists()
            assert landscape.root.is_dir()
            # Should be in system temp
            assert str(landscape.root).startswith(tempfile.gettempdir())
        finally:
            landscape.cleanup()

    def test_creates_category_subdirectories(self):
        """Landscape creates all category directories on init."""
        landscape = RunLandscape()
        try:
            for category in ("inputs", "outputs", "config", "intermediate"):
                assert (landscape.root / category).exists()
                assert (landscape.root / category).is_dir()
        finally:
            landscape.cleanup()

    def test_uses_provided_base_path(self):
        """Landscape uses provided base_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "my_landscape"
            landscape = RunLandscape(base_path=base)
            try:
                assert landscape.root == base
                assert landscape.root.exists()
            finally:
                landscape.cleanup()


class TestRunLandscapeArtifacts:
    """Test artifact registration and path management."""

    def test_get_path_creates_plugin_directory(self):
        """get_path creates plugin subdirectory and returns file path."""
        landscape = RunLandscape()
        try:
            path = landscape.get_path("inputs", "azure_blob", "data.csv")

            assert path.parent.exists()
            assert path.parent.name == "azure_blob"
            assert path.name == "data.csv"
            assert str(path).startswith(str(landscape.root / "inputs"))
        finally:
            landscape.cleanup()

    def test_get_path_raises_for_invalid_category(self):
        """get_path raises ValueError for unknown category."""
        landscape = RunLandscape()
        try:
            with pytest.raises(ValueError, match="Invalid category"):
                landscape.get_path("invalid", "plugin", "file.txt")
        finally:
            landscape.cleanup()

    def test_register_artifact_adds_to_manifest(self):
        """register_artifact tracks file in manifest."""
        landscape = RunLandscape()
        try:
            path = landscape.get_path("inputs", "csv_source", "data.csv")
            path.write_text("id,name\n1,test\n")

            landscape.register_artifact("inputs", "csv_source", path, {"rows": 1})

            manifest = landscape.get_manifest()
            assert len(manifest["artifacts"]) == 1
            artifact = manifest["artifacts"][0]
            assert artifact["category"] == "inputs"
            assert artifact["plugin_id"] == "csv_source"
            assert artifact["size"] == path.stat().st_size
            assert "sha256" in artifact
        finally:
            landscape.cleanup()

    def test_register_plugin_tracks_config(self):
        """register_plugin stores plugin configuration."""
        landscape = RunLandscape()
        try:
            landscape.register_plugin("my_source", "azure_blob", {"container": "data"})

            manifest = landscape.get_manifest()
            assert "my_source" in manifest["plugin_registry"]
            assert manifest["plugin_registry"]["my_source"]["plugin"] == "azure_blob"
            assert manifest["plugin_registry"]["my_source"]["config"]["container"] == "data"
        finally:
            landscape.cleanup()


class TestRunLandscapeConfigCapture:
    """Test configuration capture functionality."""

    def test_save_config_captures_resolved(self):
        """save_config writes resolved config to config/resolved/."""
        landscape = RunLandscape()
        try:
            resolved = {"llm": {"model": "gpt-4"}, "prompt": "test"}
            landscape.save_config(original_paths=[], resolved=resolved)

            resolved_path = landscape.root / "config" / "resolved" / "settings.yaml"
            assert resolved_path.exists()

            import yaml
            with resolved_path.open() as f:
                saved = yaml.safe_load(f)
            assert saved == resolved
        finally:
            landscape.cleanup()

    def test_save_config_copies_original_files(self):
        """save_config copies original config files."""
        import tempfile

        landscape = RunLandscape()
        try:
            # Create a temp config file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write("key: value\n")
                original_path = Path(f.name)

            try:
                landscape.save_config(original_paths=[original_path], resolved={})

                copied = landscape.root / "config" / "original" / original_path.name
                assert copied.exists()
                assert copied.read_text() == "key: value\n"
            finally:
                original_path.unlink()
        finally:
            landscape.cleanup()

    def test_save_config_registers_artifacts(self):
        """save_config registers config files as artifacts."""
        landscape = RunLandscape()
        try:
            landscape.save_config(original_paths=[], resolved={"key": "value"})

            manifest = landscape.get_manifest()
            config_artifacts = [a for a in manifest["artifacts"] if a["category"] == "config"]
            assert len(config_artifacts) >= 1
        finally:
            landscape.cleanup()


class TestRunLandscapeContextManager:
    """Test context manager and context variable integration."""

    def test_context_manager_sets_current_landscape(self):
        """Context manager sets landscape as current."""
        from elspeth.core.landscape import get_current_landscape

        assert get_current_landscape() is None

        with RunLandscape() as landscape:
            current = get_current_landscape()
            assert current is landscape

        assert get_current_landscape() is None

    def test_context_manager_cleans_up_on_exit(self):
        """Context manager cleans up landscape on exit."""
        with RunLandscape() as landscape:
            root = landscape.root
            assert root.exists()

        assert not root.exists()

    def test_context_manager_cleans_up_on_exception(self):
        """Context manager cleans up even if exception raised."""
        root = None
        try:
            with RunLandscape() as landscape:
                root = landscape.root
                raise ValueError("test error")
        except ValueError:
            pass

        assert root is not None
        assert not root.exists()

    def test_persist_keeps_directory_after_exit(self):
        """persist=True keeps directory after context exit."""
        with RunLandscape(persist=True) as landscape:
            root = landscape.root

        try:
            assert root.exists()
        finally:
            shutil.rmtree(root)


class TestRunLandscapeLLMLogging:
    """Test LLM call logging functionality."""

    def test_log_llm_call_writes_json(self):
        """log_llm_call writes request/response to intermediate/llm_calls/."""
        landscape = RunLandscape(capture_llm_calls=True)
        try:
            request = {"system": "You are helpful", "user": "Hello"}
            response = {"content": "Hi there!", "tokens": 10}

            landscape.log_llm_call("call_001", request, response, {"row_id": 1})

            call_path = landscape.root / "intermediate" / "llm_calls" / "call_001.json"
            assert call_path.exists()

            import json
            with call_path.open() as f:
                saved = json.load(f)

            assert saved["request"] == request
            assert saved["response"] == response
            assert saved["metadata"]["row_id"] == 1
        finally:
            landscape.cleanup()

    def test_log_llm_call_skipped_when_disabled(self):
        """log_llm_call does nothing when capture_llm_calls=False."""
        landscape = RunLandscape(capture_llm_calls=False)
        try:
            landscape.log_llm_call("call_001", {}, {}, {})

            calls_dir = landscape.root / "intermediate" / "llm_calls"
            assert not calls_dir.exists() or not list(calls_dir.iterdir())
        finally:
            landscape.cleanup()

    def test_log_llm_call_generates_unique_id(self):
        """log_llm_call can auto-generate call ID."""
        landscape = RunLandscape(capture_llm_calls=True)
        try:
            call_id = landscape.log_llm_call(None, {"user": "test"}, {"content": "ok"}, {})

            assert call_id is not None
            call_path = landscape.root / "intermediate" / "llm_calls" / f"{call_id}.json"
            assert call_path.exists()
        finally:
            landscape.cleanup()


class TestRunLandscapeManifest:
    """Test manifest file generation."""

    def test_write_manifest_creates_json_file(self):
        """write_manifest creates manifest.json at root."""
        landscape = RunLandscape()
        try:
            # Register some artifacts
            path = landscape.get_path("inputs", "test", "data.csv")
            path.write_text("a,b\n1,2\n")
            landscape.register_artifact("inputs", "test", path, {})

            landscape.write_manifest()

            manifest_path = landscape.root / "manifest.json"
            assert manifest_path.exists()

            import json
            with manifest_path.open() as f:
                saved = json.load(f)

            assert "created_at" in saved
            assert "artifacts" in saved
            assert len(saved["artifacts"]) == 1
        finally:
            landscape.cleanup()
