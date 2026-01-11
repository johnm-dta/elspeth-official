"""Tests for ArchiveBundleSink landscape integration."""

import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

from elspeth.core.landscape import RunLandscape
from elspeth.core.security.signing import verify_signature


class TestArchiveBundleLandscape:
    """Test ArchiveBundleSink includes landscape in archive."""

    @pytest.fixture
    def signing_key(self, monkeypatch):
        """Provide signing key via environment."""
        monkeypatch.setenv("DMP_ARCHIVE_SIGNING_KEY", "test-signing-key-12345")
        return "test-signing-key-12345"

    def test_archive_includes_landscape_artifacts(self, signing_key):
        """ArchiveBundleSink includes landscape artifacts in archive."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            with RunLandscape() as landscape:
                # Add some artifacts to landscape
                input_path = landscape.get_path("inputs", "test_source", "data.csv")
                input_path.write_text("id,value\n1,100\n")
                landscape.register_artifact("inputs", "test_source", input_path, {"rows": 1})

                # Create an LLM call
                landscape.log_llm_call("call_001", {"user": "Hello"}, {"content": "Hi"}, {})

                sink = ArchiveBundleSink(
                    base_path=base_path,
                    archive_name="test_bundle",
                    timestamped=False,
                    project_root=Path(tmpdir),  # Empty, no project files
                )

                results = {"results": [{"id": 1}]}
                sink.write(results, metadata={})

                # Check archive was created
                archive_path = sink._last_archive
                assert archive_path is not None
                assert archive_path.exists()

                # Check landscape artifacts are in archive
                with ZipFile(archive_path) as zf:
                    names = zf.namelist()
                    # Should have landscape directory
                    landscape_entries = [n for n in names if n.startswith("landscape/")]
                    assert len(landscape_entries) > 0

                    # Should include inputs
                    assert any("inputs/test_source/data.csv" in n for n in landscape_entries)

                    # Should include LLM calls
                    assert any("llm_calls" in n for n in landscape_entries)

    def test_archive_works_without_landscape(self, signing_key):
        """ArchiveBundleSink works when no landscape active."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            # Create a dummy file to archive
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            (project_root / "test.py").write_text("# test")

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="test_bundle",
                timestamped=False,
                project_root=project_root,
            )

            results = {"results": []}
            sink.write(results, metadata={})

            # Archive should exist
            assert sink._last_archive is not None
            assert sink._last_archive.exists()

    def test_archive_includes_landscape_manifest(self, signing_key):
        """ArchiveBundleSink includes landscape manifest.json in archive."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            with RunLandscape() as landscape:
                # Register a plugin
                landscape.register_plugin("my_source", "csv", {"path": "/data"})

                sink = ArchiveBundleSink(
                    base_path=base_path,
                    archive_name="test_bundle",
                    timestamped=False,
                    project_root=Path(tmpdir),
                )

                results = {"results": []}
                sink.write(results, metadata={})

                archive_path = sink._last_archive
                assert archive_path is not None

                with ZipFile(archive_path) as zf:
                    names = zf.namelist()
                    # Should include landscape manifest
                    assert "landscape/manifest.json" in names

                    # Read and verify manifest
                    manifest_data = json.loads(zf.read("landscape/manifest.json"))
                    assert "my_source" in manifest_data.get("plugin_registry", {})

    def test_archive_landscape_content_integrity(self, signing_key):
        """Verify archived landscape content matches original data exactly."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            # Define test data with specific content we can verify
            input_csv_content = "id,name,score\n1,alice,95.5\n2,bob,87.3\n3,carol,91.0\n"
            llm_request = {"system_prompt": "You are a scorer.", "user_prompt": "Score this: alice"}
            llm_response = {"content": "Score: 95.5", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            llm_metadata = {"row_id": 1, "cycle": "test_cycle"}
            config_resolved = {"llm": {"model": "test-model", "temperature": 0.7}, "suite": {"name": "test"}}
            config_paths = [Path(tmpdir) / "settings.yaml"]
            config_paths[0].write_text("llm:\n  model: test-model\n")

            with RunLandscape() as landscape:
                # 1. Add input data
                input_path = landscape.get_path("inputs", "primary_source", "data.csv")
                input_path.write_text(input_csv_content)
                landscape.register_artifact("inputs", "primary_source", input_path, {
                    "rows": 3,
                    "columns": ["id", "name", "score"],
                    "source": "test_input.csv",
                })

                # 2. Log LLM call
                landscape.log_llm_call("call_001", llm_request, llm_response, llm_metadata)

                # 3. Save config
                landscape.save_config(config_paths, config_resolved)

                # 4. Register plugin
                landscape.register_plugin("primary_source", "csv", {"path": "/data/input.csv"})

                # Create archive
                sink = ArchiveBundleSink(
                    base_path=base_path,
                    archive_name="integrity_test",
                    timestamped=False,
                    project_root=Path(tmpdir),
                )
                sink.write({"results": [{"id": 1}]}, metadata={})

                archive_path = sink._last_archive
                assert archive_path is not None

                # === VERIFY ARCHIVE CONTENTS ===
                with ZipFile(archive_path) as zf:
                    names = zf.namelist()

                    # --- Verify input data ---
                    input_arcname = "landscape/inputs/primary_source/data.csv"
                    assert input_arcname in names, f"Missing input file. Archive contains: {names}"
                    archived_input = zf.read(input_arcname).decode("utf-8")
                    assert archived_input == input_csv_content, "Input CSV content mismatch"

                    # --- Verify LLM call ---
                    llm_files = [n for n in names if "landscape/intermediate/llm_calls/" in n and n.endswith(".json")]
                    assert len(llm_files) == 1, f"Expected 1 LLM call file, got {len(llm_files)}"
                    llm_call_data = json.loads(zf.read(llm_files[0]))
                    assert llm_call_data["request"]["system_prompt"] == llm_request["system_prompt"]
                    assert llm_call_data["request"]["user_prompt"] == llm_request["user_prompt"]
                    assert llm_call_data["response"]["content"] == llm_response["content"]
                    assert llm_call_data["metadata"]["row_id"] == llm_metadata["row_id"]
                    assert "timestamp" in llm_call_data

                    # --- Verify resolved config ---
                    resolved_config_path = "landscape/config/resolved/settings.yaml"
                    assert resolved_config_path in names, f"Missing resolved config. Archive contains: {names}"
                    archived_config = yaml.safe_load(zf.read(resolved_config_path))
                    assert archived_config["llm"]["model"] == "test-model"
                    assert archived_config["llm"]["temperature"] == 0.7

                    # --- Verify original config copied ---
                    original_configs = [n for n in names if "landscape/config/original/" in n]
                    assert len(original_configs) >= 1, "Original config should be archived"

                    # --- Verify manifest ---
                    manifest_path = "landscape/manifest.json"
                    assert manifest_path in names
                    manifest = json.loads(zf.read(manifest_path))

                    # Check manifest structure
                    assert "created_at" in manifest
                    assert "artifacts" in manifest
                    assert "plugin_registry" in manifest

                    # Check artifact registered
                    input_artifacts = [a for a in manifest["artifacts"] if a["category"] == "inputs"]
                    assert len(input_artifacts) == 1
                    assert input_artifacts[0]["plugin_id"] == "primary_source"
                    assert input_artifacts[0]["metadata"]["rows"] == 3

                    # Check plugin registered
                    assert "primary_source" in manifest["plugin_registry"]
                    assert manifest["plugin_registry"]["primary_source"]["plugin"] == "csv"

    def test_archive_preserves_landscape_directory_structure(self, signing_key):
        """Verify archive preserves the full landscape directory hierarchy."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            with RunLandscape() as landscape:
                # Create files in various landscape categories
                categories = {
                    "inputs": [("source_a", "data.csv"), ("source_b", "data.json")],
                    "outputs": [("sink_a", "results.csv")],
                    "intermediate": [("transform_1", "temp.json")],
                }

                for category, files in categories.items():
                    for name, filename in files:
                        path = landscape.get_path(category, name, filename)
                        path.write_text(f"content for {category}/{name}/{filename}")
                        landscape.register_artifact(category, name, path, {})

                # Save config to ensure config directory exists
                config_file = Path(tmpdir) / "test_config.yaml"
                config_file.write_text("test: true\n")
                landscape.save_config([config_file], {"test": True})

                # Also add multiple LLM calls
                for i in range(3):
                    landscape.log_llm_call(
                        f"call_{i:03d}",
                        {"system_prompt": "sys", "user_prompt": f"user {i}"},
                        {"content": f"response {i}"},
                        {"index": i},
                    )

                sink = ArchiveBundleSink(
                    base_path=base_path,
                    archive_name="structure_test",
                    timestamped=False,
                    project_root=Path(tmpdir),
                )
                sink.write({"results": []}, metadata={})

                with ZipFile(sink._last_archive) as zf:
                    names = zf.namelist()

                    # Verify all category directories present
                    assert any("landscape/inputs/" in n for n in names)
                    assert any("landscape/outputs/" in n for n in names)
                    assert any("landscape/intermediate/" in n for n in names)
                    assert any("landscape/config/" in n for n in names)

                    # Verify specific files
                    assert "landscape/inputs/source_a/data.csv" in names
                    assert "landscape/inputs/source_b/data.json" in names
                    assert "landscape/outputs/sink_a/results.csv" in names
                    assert "landscape/intermediate/transform_1/temp.json" in names

                    # Verify LLM calls (3 of them)
                    llm_files = [n for n in names if "llm_calls" in n and n.endswith(".json")]
                    assert len(llm_files) == 3, f"Expected 3 LLM call files, got {len(llm_files)}"

                    # Verify manifest present
                    assert "landscape/manifest.json" in names


class TestArchiveBundleSignature:
    """Test ArchiveBundleSink signature verification with dummy signing key."""

    DUMMY_SIGNING_KEY = "test-dummy-signing-key-for-verification-12345"

    @pytest.fixture
    def signing_key(self, monkeypatch):
        """Provide signing key via environment."""
        monkeypatch.setenv("DMP_ARCHIVE_SIGNING_KEY", self.DUMMY_SIGNING_KEY)
        return self.DUMMY_SIGNING_KEY

    def test_archive_signature_is_valid(self, signing_key):
        """Verify archive signature can be validated with the same key."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            with RunLandscape() as landscape:
                # Add some content to make the archive non-trivial
                data_path = landscape.get_path("inputs", "test", "data.csv")
                data_path.write_text("id,value\n1,100\n2,200\n")
                landscape.register_artifact("inputs", "test", data_path, {"rows": 2})

                sink = ArchiveBundleSink(
                    base_path=base_path,
                    archive_name="signed_test",
                    timestamped=False,
                    project_root=Path(tmpdir),
                )
                sink.write({"results": [{"id": 1}]}, metadata={"test": True})

                # Get the generated files
                archive_path = sink._last_archive
                signature_path = sink._last_signature

                assert archive_path is not None
                assert signature_path is not None
                assert archive_path.exists()
                assert signature_path.exists()

                # Read signature file
                signature_data = json.loads(signature_path.read_text())
                assert "archive_signature" in signature_data
                assert "manifest_signature" in signature_data
                assert signature_data["algorithm"] == "hmac-sha256"

                # Verify archive signature using the same key
                archive_bytes = archive_path.read_bytes()
                is_valid = verify_signature(
                    archive_bytes,
                    signature_data["archive_signature"],
                    signing_key,
                    algorithm="hmac-sha256",
                )
                assert is_valid, "Archive signature should be valid with correct key"

    def test_archive_signature_fails_with_wrong_key(self, signing_key):
        """Verify archive signature fails validation with wrong key."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="signed_test",
                timestamped=False,
                project_root=Path(tmpdir),
            )

            # Create a dummy file so archive isn't empty
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            (project_root / "test.py").write_text("# test")
            sink.project_root = project_root

            sink.write({"results": []}, metadata={})

            archive_path = sink._last_archive
            signature_path = sink._last_signature

            signature_data = json.loads(signature_path.read_text())
            archive_bytes = archive_path.read_bytes()

            # Verify with WRONG key - should fail
            wrong_key = "completely-different-wrong-key"
            is_valid = verify_signature(
                archive_bytes,
                signature_data["archive_signature"],
                wrong_key,
                algorithm="hmac-sha256",
            )
            assert not is_valid, "Archive signature should NOT be valid with wrong key"

    def test_manifest_signature_is_valid(self, signing_key):
        """Verify manifest signature can be validated independently."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            with RunLandscape() as landscape:
                landscape.log_llm_call("call_1", {"system_prompt": "s", "user_prompt": "u"}, {"content": "r"}, {})

                sink = ArchiveBundleSink(
                    base_path=base_path,
                    archive_name="manifest_test",
                    timestamped=False,
                    project_root=Path(tmpdir),
                )
                sink.write({"results": []}, metadata={})

                signature_path = sink._last_signature
                manifest_path = sink._last_manifest

                signature_data = json.loads(signature_path.read_text())
                manifest_bytes = manifest_path.read_bytes()

                # Verify manifest signature
                is_valid = verify_signature(
                    manifest_bytes,
                    signature_data["manifest_signature"],
                    signing_key,
                    algorithm="hmac-sha256",
                )
                assert is_valid, "Manifest signature should be valid"

    def test_tampered_archive_fails_verification(self, signing_key):
        """Verify that modifying archive content invalidates signature."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            # Create project with a file
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            (project_root / "code.py").write_text("print('hello')")

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="tamper_test",
                timestamped=False,
                project_root=project_root,
            )
            sink.write({"results": [{"data": "original"}]}, metadata={})

            archive_path = sink._last_archive
            signature_path = sink._last_signature

            signature_data = json.loads(signature_path.read_text())
            original_signature = signature_data["archive_signature"]

            # Read original archive, tamper with it (append bytes)
            original_bytes = archive_path.read_bytes()
            tampered_bytes = original_bytes + b"TAMPERED"

            # Original signature should NOT validate tampered content
            is_valid = verify_signature(
                tampered_bytes,
                original_signature,
                signing_key,
                algorithm="hmac-sha256",
            )
            assert not is_valid, "Tampered archive should fail signature verification"

            # But original should still validate
            is_valid_original = verify_signature(
                original_bytes,
                original_signature,
                signing_key,
                algorithm="hmac-sha256",
            )
            assert is_valid_original, "Original archive should still be valid"

    def test_signature_with_sha512_algorithm(self, monkeypatch):
        """Verify signing works with SHA-512 algorithm."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        sha512_key = "sha512-test-key-longer-for-security"
        monkeypatch.setenv("DMP_ARCHIVE_SIGNING_KEY", sha512_key)

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            (project_root / "app.py").write_text("# app")

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="sha512_test",
                timestamped=False,
                project_root=project_root,
                algorithm="hmac-sha512",  # Use SHA-512
            )
            sink.write({"results": []}, metadata={})

            signature_path = sink._last_signature
            archive_path = sink._last_archive

            signature_data = json.loads(signature_path.read_text())
            assert signature_data["algorithm"] == "hmac-sha512"

            # Verify with SHA-512
            archive_bytes = archive_path.read_bytes()
            is_valid = verify_signature(
                archive_bytes,
                signature_data["archive_signature"],
                sha512_key,
                algorithm="hmac-sha512",
            )
            assert is_valid, "SHA-512 signature should be valid"


class TestArchiveBundleCollectArtifacts:
    """Tests for collect_artifacts() method."""

    @pytest.fixture
    def signing_key(self, monkeypatch):
        """Provide signing key via environment."""
        monkeypatch.setenv("DMP_ARCHIVE_SIGNING_KEY", "test-signing-key-12345")
        return "test-signing-key-12345"

    def test_collect_artifacts_returns_archive_manifest_signature(self, signing_key):
        """collect_artifacts returns all three artifact types."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            # Create a file to ensure archive is created
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            (project_root / "main.py").write_text("# main")

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="collect_test",
                timestamped=False,
                project_root=project_root,
            )

            sink.write({"results": [{"id": 1}]}, metadata={})

            # Collect artifacts
            artifacts = sink.collect_artifacts()

            assert "archive" in artifacts
            assert "manifest" in artifacts
            assert "signature" in artifacts

            # Verify artifact properties
            assert artifacts["archive"].type == "file/zip"
            assert artifacts["archive"].persist is True
            assert Path(artifacts["archive"].path).exists()

            assert artifacts["manifest"].type == "file/json"
            assert artifacts["signature"].type == "file/json"

    def test_collect_artifacts_returns_empty_when_no_archive(self, signing_key):
        """collect_artifacts returns empty dict before write() called."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = ArchiveBundleSink(
                base_path=Path(tmpdir),
                archive_name="no_write",
            )

            # Don't call write()
            artifacts = sink.collect_artifacts()

            assert artifacts == {}

    def test_collect_artifacts_includes_algorithm_metadata(self, signing_key):
        """Signature artifact includes algorithm in metadata."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            (project_root / "app.py").write_text("# app")

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="meta_test",
                timestamped=False,
                project_root=project_root,
                algorithm="hmac-sha256",
            )

            sink.write({"results": []}, metadata={})
            artifacts = sink.collect_artifacts()

            assert artifacts["signature"].metadata["algorithm"] == "hmac-sha256"


class TestArchiveBundleExtraPaths:
    """Tests for extra_paths configuration."""

    @pytest.fixture
    def signing_key(self, monkeypatch):
        """Provide signing key via environment."""
        monkeypatch.setenv("DMP_ARCHIVE_SIGNING_KEY", "test-key")
        return "test-key"

    def test_extra_paths_includes_files(self, signing_key):
        """extra_paths includes specified files in archive."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            # Create extra files outside project
            extra_dir = Path(tmpdir) / "extra_data"
            extra_dir.mkdir()
            (extra_dir / "config.json").write_text('{"key": "value"}')
            (extra_dir / "data.csv").write_text("id,name\n1,alice\n")

            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            (project_root / "main.py").write_text("# main")

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="extra_test",
                timestamped=False,
                project_root=project_root,
                extra_paths=[str(extra_dir)],
            )

            sink.write({"results": []}, metadata={})

            with ZipFile(sink._last_archive) as zf:
                names = zf.namelist()

                # Extra files should be under inputs/extra/
                extra_entries = [n for n in names if "inputs/extra/" in n]
                assert len(extra_entries) >= 2
                assert any("config.json" in n for n in extra_entries)
                assert any("data.csv" in n for n in extra_entries)

    def test_extra_paths_handles_single_file(self, signing_key):
        """extra_paths handles a single file path."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            # Create a single extra file
            extra_file = Path(tmpdir) / "special_config.yaml"
            extra_file.write_text("setting: true\n")

            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            (project_root / "app.py").write_text("# app")

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="single_extra",
                timestamped=False,
                project_root=project_root,
                extra_paths=[str(extra_file)],
            )

            sink.write({"results": []}, metadata={})

            with ZipFile(sink._last_archive) as zf:
                names = zf.namelist()
                assert any("special_config.yaml" in n for n in names)


class TestArchiveBundleMetadataDatasetKey:
    """Tests for metadata_dataset_key configuration."""

    @pytest.fixture
    def signing_key(self, monkeypatch):
        """Provide signing key via environment."""
        monkeypatch.setenv("DMP_ARCHIVE_SIGNING_KEY", "test-key")
        return "test-key"

    def test_metadata_dataset_key_includes_paths(self, signing_key):
        """Paths from metadata[dataset_key] are included in archive."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            # Create dataset files
            dataset1 = Path(tmpdir) / "train.csv"
            dataset1.write_text("id,label\n1,positive\n2,negative\n")

            dataset2 = Path(tmpdir) / "test.csv"
            dataset2.write_text("id,label\n3,positive\n")

            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            (project_root / "model.py").write_text("# model")

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="dataset_test",
                timestamped=False,
                project_root=project_root,
                metadata_dataset_key="dataset_paths",
            )

            # Pass dataset paths in metadata
            metadata = {
                "dataset_paths": [str(dataset1), str(dataset2)],
            }
            sink.write({"results": []}, metadata=metadata)

            with ZipFile(sink._last_archive) as zf:
                names = zf.namelist()

                # Datasets should be under inputs/
                input_entries = [n for n in names if "inputs/" in n and "landscape" not in n]
                assert any("train.csv" in n for n in input_entries)
                assert any("test.csv" in n for n in input_entries)

    def test_custom_metadata_dataset_key(self, signing_key):
        """Custom metadata_dataset_key name works."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            data_file = Path(tmpdir) / "data.json"
            data_file.write_text('{"items": []}')

            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            (project_root / "run.py").write_text("# run")

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="custom_key_test",
                timestamped=False,
                project_root=project_root,
                metadata_dataset_key="input_files",  # Custom key
            )

            metadata = {
                "input_files": [str(data_file)],
                "dataset_paths": ["/should/be/ignored"],  # Default key, should be ignored
            }
            sink.write({"results": []}, metadata=metadata)

            with ZipFile(sink._last_archive) as zf:
                names = zf.namelist()
                assert any("data.json" in n for n in names)

    def test_disabled_metadata_dataset_key(self, signing_key):
        """Setting metadata_dataset_key=None disables feature."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            data_file = Path(tmpdir) / "data.csv"
            data_file.write_text("a,b\n1,2\n")

            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            (project_root / "main.py").write_text("# main")

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="disabled_key",
                timestamped=False,
                project_root=project_root,
                metadata_dataset_key=None,  # Disabled
            )

            metadata = {
                "dataset_paths": [str(data_file)],
            }
            sink.write({"results": []}, metadata=metadata)

            with ZipFile(sink._last_archive) as zf:
                names = zf.namelist()
                # data.csv should NOT be in inputs/ since feature is disabled
                input_csvs = [n for n in names if "inputs/" in n and "data.csv" in n]
                assert len(input_csvs) == 0


class TestArchiveBundleEmptyArchive:
    """Tests for empty archive handling."""

    @pytest.fixture
    def signing_key(self, monkeypatch):
        """Provide signing key via environment."""
        monkeypatch.setenv("DMP_ARCHIVE_SIGNING_KEY", "test-key")
        return "test-key"

    def test_empty_project_logs_warning_no_archive(self, signing_key, caplog):
        """Empty project with no files logs warning and skips archive."""
        import logging

        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "output"
            base_path.mkdir()

            # Empty project directory
            project_root = Path(tmpdir) / "empty_project"
            project_root.mkdir()

            sink = ArchiveBundleSink(
                base_path=base_path,
                archive_name="empty_test",
                timestamped=False,
                project_root=project_root,
            )

            with caplog.at_level(logging.WARNING):
                sink.write({"results": []}, metadata={})

            # Should log warning
            assert "no files to archive" in caplog.text.lower()

            # Should not create archive
            assert sink._last_archive is None


class TestArchiveBundleValidation:
    """Tests for configuration validation."""

    def test_invalid_on_error_raises(self, tmp_path):
        """Invalid on_error value raises ValueError."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        with pytest.raises(ValueError, match="on_error must be 'abort'"):
            ArchiveBundleSink(
                base_path=tmp_path,
                on_error="ignore",
            )

    def test_missing_key_raises_on_write(self, tmp_path, monkeypatch):
        """Missing signing key raises error on write."""
        from elspeth.plugins.outputs.archive_bundle import ArchiveBundleSink

        monkeypatch.delenv("DMP_ARCHIVE_SIGNING_KEY", raising=False)

        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "app.py").write_text("# app")

        sink = ArchiveBundleSink(
            base_path=tmp_path,
            project_root=project_root,
            key=None,
            key_env="DMP_ARCHIVE_SIGNING_KEY",
        )

        with pytest.raises(ValueError, match="Signing key not provided"):
            sink.write({"results": []}, metadata={})
