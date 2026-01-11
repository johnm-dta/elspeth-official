"""Tests for AzureArchiveBundleSink split archive functionality."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest


class TestAzureArchiveBundleSplit:
    """Test split archive functionality for AzureArchiveBundleSink."""

    @pytest.fixture
    def mock_keyvault(self):
        """Mock Azure Key Vault clients."""
        with patch(
            "elspeth.plugins.outputs.azure_archive_bundle._get_keyvault_client"
        ) as mock_kv, patch(
            "elspeth.plugins.outputs.azure_archive_bundle._get_crypto_client"
        ) as mock_crypto, patch(
            "elspeth.plugins.outputs.azure_archive_bundle.AzureArchiveBundleSink._get_public_key_pem"
        ) as mock_pem:
            # Mock key client
            mock_key_client = MagicMock()
            mock_key_client.get_key.return_value = MagicMock()
            mock_kv.return_value = mock_key_client

            # Mock crypto client with a simple signature
            mock_crypto_client = MagicMock()
            mock_result = MagicMock()
            mock_result.signature = b"mock_signature_bytes"
            mock_crypto_client.sign.return_value = mock_result
            mock_crypto.return_value = mock_crypto_client

            # Mock public key PEM
            mock_pem.return_value = "-----BEGIN PUBLIC KEY-----\nMOCK\n-----END PUBLIC KEY-----"

            yield {"key_client": mock_key_client, "crypto_client": mock_crypto_client}

    @pytest.fixture
    def sink_factory(self, mock_keyvault, tmp_path):
        """Factory to create configured sinks."""

        def create_sink(**kwargs):
            from elspeth.plugins.outputs.azure_archive_bundle import (
                AzureArchiveBundleSink,
            )

            defaults = {
                "base_path": tmp_path / "output",
                "vault_url": "https://test-vault.vault.azure.net",
                "archive_name": "test_bundle",
                "timestamped": False,
                "project_root": tmp_path / "project",
            }
            defaults.update(kwargs)
            defaults["base_path"].mkdir(parents=True, exist_ok=True)
            defaults["project_root"].mkdir(parents=True, exist_ok=True)
            return AzureArchiveBundleSink(**defaults)

        return create_sink

    def test_needs_splitting_returns_false_when_disabled(self, sink_factory, tmp_path):
        """_needs_splitting returns False when max_part_size_mb is 0."""
        sink = sink_factory(max_part_size_mb=0)

        # Create a test file
        test_file = tmp_path / "output" / "test.zip"
        test_file.write_bytes(b"x" * 100_000_000)  # 100MB

        assert sink._needs_splitting(test_file) is False

    def test_needs_splitting_returns_false_for_small_files(
        self, sink_factory, tmp_path
    ):
        """_needs_splitting returns False for files under limit."""
        sink = sink_factory(max_part_size_mb=50)

        # Create a small file (1MB)
        test_file = tmp_path / "output" / "small.zip"
        test_file.write_bytes(b"x" * 1_000_000)

        assert sink._needs_splitting(test_file) is False

    def test_needs_splitting_returns_true_for_large_files(self, sink_factory, tmp_path):
        """_needs_splitting returns True for files over limit."""
        sink = sink_factory(max_part_size_mb=1)  # 1MB limit

        # Create a file larger than 1MB
        test_file = tmp_path / "output" / "large.zip"
        test_file.write_bytes(b"x" * 2_000_000)  # 2MB

        assert sink._needs_splitting(test_file) is True

    @pytest.mark.skipif(
        not shutil.which("zip"), reason="zip command not available"
    )
    def test_split_archive_creates_parts(self, sink_factory, tmp_path):
        """_split_archive creates multiple part files."""
        sink = sink_factory(max_part_size_mb=1)  # 1MB parts

        # Create a large zip file (need real zip for split to work)
        project_dir = tmp_path / "project"
        large_file = project_dir / "large_data.bin"
        large_file.write_bytes(os.urandom(3_000_000))  # 3MB of random data

        # Create initial archive
        archive_path = tmp_path / "output" / "test_bundle.zip"
        with ZipFile(archive_path, "w") as zf:
            zf.write(large_file, "large_data.bin")

        # Verify it's large enough to split
        assert archive_path.stat().st_size > 1_000_000

        # Split the archive
        parts = sink._split_archive(archive_path)

        # Should have multiple parts
        assert len(parts) > 1

        # Original should be deleted
        assert not archive_path.exists()

        # All parts should exist
        for part in parts:
            assert part.exists()

        # Last part should be .zip
        assert parts[-1].suffix == ".zip"

        # Other parts should be .z01, .z02, etc.
        for part in parts[:-1]:
            assert part.suffix.startswith(".z")

    def test_write_single_archive_no_split(self, sink_factory, tmp_path):
        """write() creates single archive when under size limit."""
        project_dir = tmp_path / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "config.yaml").write_text("test: true")  # Use yaml to match patterns

        sink = sink_factory(max_part_size_mb=50)  # 50MB limit

        sink.write({"results": [{"id": 1}]}, metadata={})

        # Should have single archive
        assert len(sink._archive_parts) == 1
        assert sink._last_archive is not None
        assert sink._last_archive.exists()

        # Manifest should have "archive" field (not "archive_parts")
        manifest_path = tmp_path / "output" / "test_bundle.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert "archive" in manifest
        assert "archive_parts" not in manifest

    @pytest.mark.skipif(
        not shutil.which("zip"), reason="zip command not available"
    )
    def test_write_split_archive_manifest_format(self, sink_factory, tmp_path):
        """write() creates manifest with archive_parts for split archives."""
        project_dir = tmp_path / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        # Create large file to force splitting (use .yaml to match default patterns)
        large_file = project_dir / "large.yaml"
        large_file.write_bytes(os.urandom(3_000_000))  # 3MB

        sink = sink_factory(max_part_size_mb=1)  # 1MB limit

        sink.write({"results": []}, metadata={})

        # Should have multiple parts
        assert len(sink._archive_parts) > 1

        # Manifest should list all parts
        manifest_path = tmp_path / "output" / "test_bundle.manifest.json"
        manifest = json.loads(manifest_path.read_text())

        assert manifest.get("split_archive") is True
        assert "archive_parts" in manifest
        assert len(manifest["archive_parts"]) == len(sink._archive_parts)

        # Each part should have name, size, sha256
        for part_info in manifest["archive_parts"]:
            assert "name" in part_info
            assert "size" in part_info
            assert "sha256" in part_info
            assert len(part_info["sha256"]) == 64  # SHA256 hex length

    @pytest.mark.skipif(
        not shutil.which("zip"), reason="zip command not available"
    )
    def test_write_split_archive_signature_format(self, sink_factory, tmp_path):
        """write() creates signature with per-part signatures for split archives."""
        project_dir = tmp_path / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        large_file = project_dir / "large.yaml"
        large_file.write_bytes(os.urandom(3_000_000))

        sink = sink_factory(max_part_size_mb=1)

        sink.write({"results": []}, metadata={})

        signature_path = tmp_path / "output" / "test_bundle.signature.json"
        signature = json.loads(signature_path.read_text())

        assert signature.get("split_archive") is True
        assert "archive_parts" in signature
        assert len(signature["archive_parts"]) == len(sink._archive_parts)

        # Each part should have name and signature
        for part_sig in signature["archive_parts"]:
            assert "name" in part_sig
            assert "signature" in part_sig

        # Should still have manifest signature
        assert "manifest_signature" in signature

    @pytest.mark.skipif(
        not shutil.which("zip"), reason="zip command not available"
    )
    def test_collect_artifacts_split_archive(self, sink_factory, tmp_path):
        """collect_artifacts returns all parts for split archives."""
        project_dir = tmp_path / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        large_file = project_dir / "large.yaml"
        large_file.write_bytes(os.urandom(3_000_000))

        sink = sink_factory(max_part_size_mb=1)
        sink.write({"results": []}, metadata={})

        artifacts = sink.collect_artifacts()

        # Should have archive_part_0, archive_part_1, etc.
        part_artifacts = [k for k in artifacts if k.startswith("archive_part_")]
        assert len(part_artifacts) == len(sink._archive_parts)

        # Should have manifest and signature
        assert "manifest" in artifacts
        assert "signature" in artifacts

        # Should have "archive" artifact with split metadata for consumers
        assert "archive" in artifacts
        archive_artifact = artifacts["archive"]
        assert archive_artifact.metadata.get("split_archive") is True
        assert "all_parts" in archive_artifact.metadata
        assert len(archive_artifact.metadata["all_parts"]) == len(sink._archive_parts)

        # Each part artifact should have correct metadata
        for i, key in enumerate(sorted(part_artifacts)):
            artifact = artifacts[key]
            assert artifact.metadata.get("part_index") == i
            assert Path(artifact.path).exists()

    def test_collect_artifacts_single_archive(self, sink_factory, tmp_path):
        """collect_artifacts returns single archive artifact when not split."""
        project_dir = tmp_path / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "config.yaml").write_text("test: true")  # Use yaml to match patterns

        sink = sink_factory(max_part_size_mb=50)
        sink.write({"results": []}, metadata={})

        artifacts = sink.collect_artifacts()

        # Should have single "archive" artifact (not parts)
        assert "archive" in artifacts
        part_artifacts = [k for k in artifacts if k.startswith("archive_part_")]
        assert len(part_artifacts) == 0

    def test_schema_includes_max_part_size_mb(self):
        """Schema includes max_part_size_mb property."""
        from elspeth.plugins.outputs.azure_archive_bundle import (
            AZURE_ARCHIVE_BUNDLE_SCHEMA,
        )

        props = AZURE_ARCHIVE_BUNDLE_SCHEMA["properties"]
        assert "max_part_size_mb" in props
        assert props["max_part_size_mb"]["type"] == "integer"
        assert props["max_part_size_mb"]["minimum"] == 0

    def test_default_max_part_size(self, sink_factory):
        """Default max_part_size_mb is 20 (under Azure DevOps ~25MB limit)."""
        sink = sink_factory()
        assert sink.max_part_size_mb == 20

    def test_split_archive_without_zip_command_raises(self, sink_factory, tmp_path):
        """_split_archive raises RuntimeError when zip command not found."""
        sink = sink_factory(max_part_size_mb=1)

        archive_path = tmp_path / "output" / "test.zip"
        archive_path.write_bytes(b"PK" + b"\x00" * 100)  # Minimal zip header

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="zip command not found"):
                sink._split_archive(archive_path)

    def test_split_archive_missing_file_raises(self, sink_factory, tmp_path):
        """_split_archive raises RuntimeError when archive file doesn't exist."""
        sink = sink_factory(max_part_size_mb=1)

        # Reference a path that doesn't exist
        archive_path = tmp_path / "output" / "nonexistent.zip"

        with pytest.raises(RuntimeError, match="Archive file not found"):
            sink._split_archive(archive_path)


class TestAzureArchiveBundleSplitReadme:
    """Test that README includes split archive instructions."""

    @pytest.fixture
    def mock_keyvault(self):
        """Mock Azure Key Vault clients."""
        with patch(
            "elspeth.plugins.outputs.azure_archive_bundle._get_keyvault_client"
        ) as mock_kv, patch(
            "elspeth.plugins.outputs.azure_archive_bundle._get_crypto_client"
        ) as mock_crypto, patch(
            "elspeth.plugins.outputs.azure_archive_bundle.AzureArchiveBundleSink._get_public_key_pem"
        ) as mock_pem:
            mock_key_client = MagicMock()
            mock_key_client.get_key.return_value = MagicMock()
            mock_kv.return_value = mock_key_client

            mock_crypto_client = MagicMock()
            mock_result = MagicMock()
            mock_result.signature = b"mock_signature"
            mock_crypto_client.sign.return_value = mock_result
            mock_crypto.return_value = mock_crypto_client

            # Mock public key PEM
            mock_pem.return_value = "-----BEGIN PUBLIC KEY-----\nMOCK\n-----END PUBLIC KEY-----"

            yield

    def test_readme_contains_split_archive_instructions(self, mock_keyvault, tmp_path):
        """Generated README explains how to handle split archives."""
        from elspeth.plugins.outputs.azure_archive_bundle import (
            AzureArchiveBundleSink,
        )

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "config.yaml").write_text("test: true")  # Use yaml to match default patterns

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sink = AzureArchiveBundleSink(
            base_path=output_dir,
            vault_url="https://test.vault.azure.net",
            project_root=project_dir,
            timestamped=False,
        )

        sink.write({"results": []}, metadata={})

        # Read the archive and check README
        with ZipFile(sink._last_archive) as zf:
            readme = zf.read("README.md").decode("utf-8")

        # Should mention split archives
        assert "Split Archive" in readme
        assert ".z01" in readme
        assert ".z02" in readme
        assert "zip -s 0" in readme  # Recombine command
        assert "7-Zip" in readme or "7z" in readme
