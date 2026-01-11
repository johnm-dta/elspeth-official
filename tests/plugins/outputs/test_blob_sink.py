"""Tests for Azure Blob Storage result sink."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from elspeth.plugins.outputs.blob import BlobResultSink


@pytest.fixture
def blob_config_file(tmp_path: Path) -> Path:
    """Create a valid blob config file."""
    config_path = tmp_path / "blob_config.yaml"
    config_path.write_text(
        """
default:
  connection_name: test-connection
  azureml_datastore_uri: azureml://datastores/test
  storage_uri: https://testaccount.blob.core.windows.net/testcontainer/testpath
"""
    )
    return config_path


@pytest.fixture
def mock_blob_service_client():
    """Create a mock Azure Blob Service client."""
    mock_service = MagicMock()
    mock_container = MagicMock()
    mock_blob = MagicMock()

    mock_service.get_container_client.return_value = mock_container
    mock_container.get_blob_client.return_value = mock_blob

    return mock_service, mock_container, mock_blob


class TestBlobResultSinkInit:
    """Tests for BlobResultSink initialization."""

    def test_init_with_valid_config(self, blob_config_file: Path):
        """Sink initializes with valid config file."""
        sink = BlobResultSink(config_path=blob_config_file, profile="default")

        assert sink.config.container_name == "testcontainer"
        assert sink.config.account_url == "https://testaccount.blob.core.windows.net"
        assert sink.filename == "results.json"
        assert sink.include_manifest is True

    def test_init_with_custom_options(self, blob_config_file: Path):
        """Sink accepts custom options."""
        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            filename="custom.json",
            include_manifest=False,
            overwrite=False,
            upload_chunk_size=1024 * 1024,
        )

        assert sink.filename == "custom.json"
        assert sink.include_manifest is False
        assert sink.overwrite is False
        assert sink.upload_chunk_size == 1024 * 1024

    def test_init_invalid_on_error(self, blob_config_file: Path):
        """Sink rejects invalid on_error values."""
        with pytest.raises(ValueError, match="on_error must be 'abort'"):
            BlobResultSink(
                config_path=blob_config_file,
                profile="default",
                on_error="ignore",
            )

    def test_init_missing_config_file(self, tmp_path: Path):
        """Sink raises error for missing config file."""
        from elspeth.datasources.blob_store import BlobConfigurationError

        with pytest.raises(BlobConfigurationError, match="not found"):
            BlobResultSink(config_path=tmp_path / "nonexistent.yaml", profile="default")

    def test_init_missing_profile(self, blob_config_file: Path):
        """Sink raises error for missing profile."""
        from elspeth.datasources.blob_store import BlobConfigurationError

        with pytest.raises(BlobConfigurationError, match="not found"):
            BlobResultSink(config_path=blob_config_file, profile="nonexistent")


class TestBlobPathResolution:
    """Tests for blob path template resolution."""

    def test_resolve_blob_name_with_template(self, blob_config_file: Path):
        """Path template is correctly formatted."""
        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            path_template="experiments/{experiment}/{date}/results.json",
        )

        context = {
            "experiment": "test-exp",
            "date": "2025-01-01",
            "timestamp": "20250101T000000Z",
            "blob_path": sink.config.blob_path,
            "container": sink.config.container_name,
            "filename": sink.filename,
        }

        blob_name = sink._resolve_blob_name(context)
        assert blob_name == "experiments/test-exp/2025-01-01/results.json"

    def test_resolve_blob_name_without_template(self, blob_config_file: Path):
        """Without template, uses config blob_path."""
        sink = BlobResultSink(config_path=blob_config_file, profile="default")

        # Without a path_template, it uses sink.config.blob_path (from the config file)
        # The config file has storage_uri with blob_path="testpath"
        # Since testpath has no extension, it appends /filename
        context = {
            "blob_path": sink.config.blob_path,
            "filename": sink.filename,
        }

        blob_name = sink._resolve_blob_name(context)
        assert blob_name == "testpath/results.json"

    def test_resolve_blob_name_appends_filename_to_directory(self, blob_config_file: Path):
        """Filename is appended when path ends with /."""
        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            path_template="experiments/output/",
            filename="data.json",
        )

        context = {"filename": sink.filename}
        blob_name = sink._resolve_blob_name(context)
        assert blob_name == "experiments/output/data.json"

    def test_resolve_blob_name_missing_placeholder(self, blob_config_file: Path):
        """Missing placeholder raises ValueError."""
        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            path_template="experiments/{missing_key}/results.json",
        )

        with pytest.raises(ValueError, match="Missing placeholder 'missing_key'"):
            sink._resolve_blob_name({})


class TestCredentialResolution:
    """Tests for credential resolution logic."""

    def test_explicit_credential_used(self, blob_config_file: Path):
        """Explicit credential is preferred."""
        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            credential="explicit-key",
        )

        result = sink._resolve_credential(sink.config)
        assert result == "explicit-key"

    def test_env_var_credential(self, blob_config_file: Path, monkeypatch):
        """Credential from environment variable."""
        monkeypatch.setenv("TEST_BLOB_KEY", "env-key")

        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            credential_env="TEST_BLOB_KEY",
        )

        result = sink._resolve_credential(sink.config)
        assert result == "env-key"

    def test_sas_token_from_config(self, tmp_path: Path):
        """SAS token from config is used."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
default:
  connection_name: test
  azureml_datastore_uri: azureml://test
  storage_uri: https://test.blob.core.windows.net/container/path
  sas_token: "sv=2021-06-08&ss=b&srt=co"
"""
        )

        sink = BlobResultSink(config_path=config_path, profile="default")
        result = sink._resolve_credential(sink.config)
        assert result == "sv=2021-06-08&ss=b&srt=co"

    @patch("elspeth.plugins.outputs.blob.DefaultAzureCredential", create=True)
    def test_default_azure_credential_fallback(self, mock_default_cred, blob_config_file: Path):
        """Falls back to DefaultAzureCredential when no other credential."""
        mock_credential_instance = MagicMock()
        mock_default_cred.return_value = mock_credential_instance

        # Patch at module level to avoid import issues
        with patch.dict("sys.modules", {"azure.identity": MagicMock(DefaultAzureCredential=mock_default_cred)}):
            sink = BlobResultSink(config_path=blob_config_file, profile="default")
            result = sink._resolve_credential(sink.config)

        assert result is not None  # Should be the DefaultAzureCredential instance


class TestBlobUpload:
    """Tests for blob upload operations."""

    def test_write_single_chunk_upload(self, blob_config_file: Path, mock_blob_service_client):
        """Small payload uses single upload."""
        mock_service, _mock_container, mock_blob = mock_blob_service_client

        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            credential="test-key",
        )
        sink._blob_service_client = mock_service

        results = {"results": [{"id": 1}], "aggregates": {"count": 1}}
        sink.write(results, metadata={"experiment": "test"})

        # Verify upload_blob was called (results + manifest = 2 calls)
        assert mock_blob.upload_blob.call_count >= 1

        # Get the first call (results upload)
        first_call_args = mock_blob.upload_blob.call_args_list[0]
        uploaded_data = first_call_args[0][0]

        # Verify uploaded data is valid JSON containing results
        parsed = json.loads(uploaded_data)
        assert parsed["results"] == [{"id": 1}]

    def test_write_chunked_upload_for_large_files(self, blob_config_file: Path, mock_blob_service_client):
        """Large payload uses chunked upload."""
        mock_service, _mock_container, mock_blob = mock_blob_service_client

        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            credential="test-key",
            upload_chunk_size=100,  # Very small chunk size for testing
        )
        sink._blob_service_client = mock_service

        # Create payload larger than chunk size
        large_results = {"results": [{"data": "x" * 200}]}
        sink.write(large_results, metadata={})

        # Verify stage_block and commit_block_list were called
        assert mock_blob.stage_block.called
        mock_blob.commit_block_list.assert_called()

    def test_write_includes_manifest(self, blob_config_file: Path, mock_blob_service_client):
        """Manifest is uploaded when include_manifest=True."""
        mock_service, _mock_container, mock_blob = mock_blob_service_client

        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            credential="test-key",
            include_manifest=True,
        )
        sink._blob_service_client = mock_service

        results = {"results": [{"id": 1}]}
        sink.write(results, metadata={"experiment": "test"})

        # Should have been called twice (results + manifest)
        assert mock_blob.upload_blob.call_count == 2

    def test_write_skips_manifest_when_disabled(self, blob_config_file: Path, mock_blob_service_client):
        """Manifest is not uploaded when include_manifest=False."""
        mock_service, _mock_container, mock_blob = mock_blob_service_client

        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            credential="test-key",
            include_manifest=False,
        )
        sink._blob_service_client = mock_service

        results = {"results": [{"id": 1}]}
        sink.write(results, metadata={})

        # Should only be called once (results only)
        assert mock_blob.upload_blob.call_count == 1


class TestManifestGeneration:
    """Tests for manifest file generation."""

    def test_build_manifest_includes_required_fields(self, blob_config_file: Path):
        """Manifest contains all required fields."""
        sink = BlobResultSink(config_path=blob_config_file, profile="default")

        results = {
            "results": [{"id": 1}, {"id": 2}],
            "aggregates": {"count": 2},
            "cost_summary": {"total": 0.01},
        }
        metadata = {"experiment": "test-exp"}
        timestamp = datetime(2025, 1, 1, tzinfo=UTC)

        manifest = sink._build_manifest(results, metadata, "output/results.json", timestamp)

        assert manifest["generated_at"] == "2025-01-01T00:00:00+00:00"
        assert manifest["blob"] == "output/results.json"
        assert manifest["container"] == "testcontainer"
        assert manifest["rows"] == 2
        assert manifest["metadata"] == {"experiment": "test-exp"}
        assert manifest["aggregates"] == {"count": 2}
        assert manifest["cost_summary"] == {"total": 0.01}

    def test_build_manifest_handles_empty_results(self, blob_config_file: Path):
        """Manifest handles empty results gracefully."""
        sink = BlobResultSink(config_path=blob_config_file, profile="default")

        results = {}
        timestamp = datetime(2025, 1, 1, tzinfo=UTC)

        manifest = sink._build_manifest(results, {}, "path", timestamp)
        assert manifest["rows"] == 0


class TestBlobMetadata:
    """Tests for Azure blob metadata handling."""

    def test_normalize_metadata_converts_to_strings(self, blob_config_file: Path):
        """Metadata values are converted to strings."""
        metadata = {"count": 42, "enabled": True, "ratio": 0.5}
        normalized = BlobResultSink._normalize_metadata(metadata)

        assert normalized == {"count": "42", "enabled": "True", "ratio": "0.5"}

    def test_normalize_metadata_handles_none_values(self, blob_config_file: Path):
        """None values become empty strings."""
        metadata = {"key": None, "other": "value"}
        normalized = BlobResultSink._normalize_metadata(metadata)

        assert normalized == {"key": "", "other": "value"}

    def test_normalize_metadata_rejects_non_string_keys(self, blob_config_file: Path):
        """Non-string keys raise ValueError."""
        with pytest.raises(ValueError, match="keys must be strings"):
            BlobResultSink._normalize_metadata({123: "value"})  # type: ignore

    def test_merge_upload_metadata_combines_sources(self, blob_config_file: Path):
        """Upload metadata is merged with base metadata."""
        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            metadata={"base_key": "base_value"},
        )

        combined = sink._merge_upload_metadata({"upload_key": "upload_value"})

        assert combined == {"base_key": "base_value", "upload_key": "upload_value"}


class TestArtifactHandling:
    """Tests for artifact input/output handling."""

    def test_prepare_artifacts_collects_from_all_sources(self, blob_config_file: Path):
        """Artifacts are collected from all upstream sources."""
        from elspeth.core.interfaces import Artifact

        sink = BlobResultSink(config_path=blob_config_file, profile="default")

        artifacts = {
            "source1": [Artifact(id="1", type="json", path="/tmp/a.json", metadata={})],
            "source2": [
                Artifact(id="2", type="json", path="/tmp/b.json", metadata={}),
                Artifact(id="3", type="json", path="/tmp/c.json", metadata={}),
            ],
        }

        sink.prepare_artifacts(artifacts)

        assert len(sink._artifact_inputs) == 3

    def test_artifact_bytes_from_path(self, blob_config_file: Path, tmp_path: Path):
        """Artifact bytes are read from path."""
        from elspeth.core.interfaces import Artifact

        test_file = tmp_path / "test.json"
        test_file.write_bytes(b'{"key": "value"}')

        artifact = Artifact(id="test", type="json", path=str(test_file), metadata={})
        result = BlobResultSink._artifact_bytes(artifact)

        assert result == b'{"key": "value"}'

    def test_artifact_bytes_from_payload_bytes(self, blob_config_file: Path):
        """Artifact bytes from bytes payload."""
        from elspeth.core.interfaces import Artifact

        artifact = Artifact(id="test", type="json", path=None, metadata={}, payload=b"raw bytes")
        result = BlobResultSink._artifact_bytes(artifact)

        assert result == b"raw bytes"

    def test_artifact_bytes_from_payload_dict(self, blob_config_file: Path):
        """Artifact bytes from dict payload are JSON serialized."""
        from elspeth.core.interfaces import Artifact

        artifact = Artifact(id="test", type="json", path=None, metadata={}, payload={"key": "value"})
        result = BlobResultSink._artifact_bytes(artifact)

        assert json.loads(result) == {"key": "value"}

    def test_artifact_bytes_missing_data_raises(self, blob_config_file: Path):
        """Missing artifact data raises ValueError."""
        from elspeth.core.interfaces import Artifact

        artifact = Artifact(id="test", type="json", path=None, metadata={}, payload=None)

        with pytest.raises(ValueError, match="missing payload data"):
            BlobResultSink._artifact_bytes(artifact)


class TestAppendSuffix:
    """Tests for blob name suffix handling."""

    def test_append_suffix_to_filename(self):
        """Suffix is appended to filename correctly."""
        result = BlobResultSink._append_suffix("results.json", 2)
        assert result == "results_2.json"

    def test_append_suffix_with_path(self):
        """Suffix is appended preserving path."""
        result = BlobResultSink._append_suffix("output/data/results.json", 3)
        assert result == "output/data/results_3.json"

    def test_append_suffix_no_extension(self):
        """Suffix works with no extension."""
        result = BlobResultSink._append_suffix("output/data", 2)
        assert result == "output/data_2"


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_write_propagates_upload_error(self, blob_config_file: Path, mock_blob_service_client):
        """Upload errors are propagated."""
        mock_service, _mock_container, mock_blob = mock_blob_service_client
        mock_blob.upload_blob.side_effect = RuntimeError("Connection failed")

        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            credential="test-key",
        )
        sink._blob_service_client = mock_service

        with pytest.raises(RuntimeError, match="Connection failed"):
            sink.write({"results": []}, metadata={})

    def test_write_clears_artifacts_on_error(self, blob_config_file: Path, mock_blob_service_client, tmp_path: Path):
        """Artifact inputs are cleared even on error."""
        from elspeth.core.interfaces import Artifact

        mock_service, _mock_container, mock_blob = mock_blob_service_client
        mock_blob.upload_blob.side_effect = RuntimeError("Upload failed")

        sink = BlobResultSink(
            config_path=blob_config_file,
            profile="default",
            credential="test-key",
        )
        sink._blob_service_client = mock_service

        # Create a real temp file for the artifact
        temp_file = tmp_path / "a.json"
        temp_file.write_text('{"test": true}')

        # Prepare some artifacts with a real file
        sink._artifact_inputs = [Artifact(id="1", type="json", path=str(temp_file), metadata={})]

        with pytest.raises(RuntimeError, match="Upload failed"):
            sink.write({"results": []}, metadata={})

        # Artifacts should be cleared
        assert sink._artifact_inputs == []


class TestSecurityLevel:
    """Tests for security level metadata propagation."""

    def test_build_upload_metadata_includes_security_level(self, blob_config_file: Path):
        """Security level from execution metadata is included."""
        sink = BlobResultSink(config_path=blob_config_file, profile="default")

        metadata = sink._build_upload_metadata(
            {"security_level": "official-sensitive"},
            None,
        )

        assert metadata["security_level"] == "official-sensitive"

    def test_artifact_security_level_takes_precedence(self, blob_config_file: Path):
        """Artifact security level overrides execution metadata."""
        from elspeth.core.interfaces import Artifact

        sink = BlobResultSink(config_path=blob_config_file, profile="default")

        artifact = Artifact(
            id="test",
            type="json",
            path=None,
            metadata={},
            security_level="secret",
        )

        metadata = sink._build_upload_metadata(
            {"security_level": "official"},
            artifact,
        )

        assert metadata["security_level"] == "secret"
