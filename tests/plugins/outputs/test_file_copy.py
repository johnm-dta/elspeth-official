"""Tests for FileCopySink."""

from __future__ import annotations

import pytest

from elspeth.core.interfaces import Artifact
from elspeth.plugins.outputs.file_copy import FileCopySink


class TestFileCopySinkInit:
    """Tests for FileCopySink initialization."""

    def test_init_with_defaults(self, tmp_path):
        """Initializes with default values."""
        sink = FileCopySink(destination=str(tmp_path / "dest.txt"))

        assert sink.destination == tmp_path / "dest.txt"
        assert sink.overwrite is True
        assert sink.on_error == "abort"

    def test_init_with_custom_options(self, tmp_path):
        """Accepts custom options."""
        sink = FileCopySink(
            destination=str(tmp_path / "custom.txt"),
            overwrite=False,
        )

        assert sink.destination == tmp_path / "custom.txt"
        assert sink.overwrite is False

    def test_init_rejects_invalid_on_error(self, tmp_path):
        """Rejects invalid on_error value."""
        with pytest.raises(ValueError, match="on_error must be 'abort'"):
            FileCopySink(destination=str(tmp_path / "dest.txt"), on_error="ignore")


class TestFileCopySinkPrepareArtifacts:
    """Tests for prepare_artifacts method."""

    def test_accepts_single_artifact(self, tmp_path):
        """Accepts a single input artifact."""
        src = tmp_path / "source.txt"
        src.write_text("content")

        sink = FileCopySink(destination=str(tmp_path / "dest.txt"))
        sink.prepare_artifacts({
            "any": [Artifact(
                id="a1",
                type="file/text",
                path=str(src),
                metadata={"content_type": "text/plain"},
            )]
        })

        assert sink._source_artifact is not None
        assert sink._source_artifact.id == "a1"

    def test_rejects_multiple_artifacts(self, tmp_path):
        """Rejects multiple input artifacts."""
        src1 = tmp_path / "source1.txt"
        src2 = tmp_path / "source2.txt"
        src1.write_text("content1")
        src2.write_text("content2")

        sink = FileCopySink(destination=str(tmp_path / "dest.txt"))

        with pytest.raises(ValueError, match="single input artifact"):
            sink.prepare_artifacts({
                "files": [
                    Artifact(id="a1", type="file/text", path=str(src1), metadata={}),
                    Artifact(id="a2", type="file/text", path=str(src2), metadata={}),
                ]
            })

    def test_handles_empty_artifacts(self, tmp_path):
        """Handles empty artifacts gracefully."""
        sink = FileCopySink(destination=str(tmp_path / "dest.txt"))
        sink.prepare_artifacts({})

        assert sink._source_artifact is None

    def test_handles_empty_list(self, tmp_path):
        """Handles empty artifact list gracefully."""
        sink = FileCopySink(destination=str(tmp_path / "dest.txt"))
        sink.prepare_artifacts({"files": []})

        assert sink._source_artifact is None

    def test_extracts_artifact_type(self, tmp_path):
        """Extracts artifact type from source."""
        src = tmp_path / "source.json"
        src.write_text('{"key": "value"}')

        sink = FileCopySink(destination=str(tmp_path / "dest.json"))
        sink.prepare_artifacts({
            "data": [Artifact(
                id="a1",
                type="file/json",
                path=str(src),
                metadata={},
            )]
        })

        assert sink._output_type == "file/json"

    def test_extracts_security_level(self, tmp_path):
        """Extracts security level from source artifact."""
        src = tmp_path / "source.txt"
        src.write_text("secret data")

        sink = FileCopySink(destination=str(tmp_path / "dest.txt"))
        sink.prepare_artifacts({
            "data": [Artifact(
                id="a1",
                type="file/text",
                path=str(src),
                metadata={},
                security_level="secret",
            )]
        })

        assert sink._security_level == "secret"


class TestFileCopySinkWrite:
    """Tests for write method."""

    def test_copies_file_to_destination(self, tmp_path):
        """Copies source file to destination."""
        src = tmp_path / "source.txt"
        src.write_text("hello world")

        dest = tmp_path / "dest.txt"
        sink = FileCopySink(destination=str(dest))
        sink.prepare_artifacts({
            "any": [Artifact(id="a1", type="file/text", path=str(src), metadata={})]
        })

        sink.write({"results": []})

        assert dest.exists()
        assert dest.read_text() == "hello world"

    def test_creates_parent_directories(self, tmp_path):
        """Creates parent directories if needed."""
        src = tmp_path / "source.txt"
        src.write_text("content")

        dest = tmp_path / "nested" / "path" / "dest.txt"
        sink = FileCopySink(destination=str(dest))
        sink.prepare_artifacts({
            "any": [Artifact(id="a1", type="file/text", path=str(src), metadata={})]
        })

        sink.write({"results": []})

        assert dest.exists()

    def test_raises_when_no_source_artifact(self, tmp_path):
        """Raises error when no source artifact prepared."""
        sink = FileCopySink(destination=str(tmp_path / "dest.txt"))

        with pytest.raises(ValueError, match="requires an input artifact"):
            sink.write({"results": []})

    def test_raises_when_source_not_found(self, tmp_path):
        """Raises error when source file doesn't exist."""
        missing_src = tmp_path / "nonexistent.txt"

        sink = FileCopySink(destination=str(tmp_path / "dest.txt"))
        sink.prepare_artifacts({
            "any": [Artifact(id="a1", type="file/text", path=str(missing_src), metadata={})]
        })

        with pytest.raises(FileNotFoundError, match="not found"):
            sink.write({"results": []})

    def test_overwrites_existing_when_enabled(self, tmp_path):
        """Overwrites existing file when overwrite=True."""
        src = tmp_path / "source.txt"
        src.write_text("new content")

        dest = tmp_path / "dest.txt"
        dest.write_text("old content")

        sink = FileCopySink(destination=str(dest), overwrite=True)
        sink.prepare_artifacts({
            "any": [Artifact(id="a1", type="file/text", path=str(src), metadata={})]
        })

        sink.write({"results": []})

        assert dest.read_text() == "new content"

    def test_raises_when_exists_and_no_overwrite(self, tmp_path):
        """Raises error when destination exists and overwrite=False."""
        src = tmp_path / "source.txt"
        src.write_text("content")

        dest = tmp_path / "dest.txt"
        dest.write_text("existing")

        sink = FileCopySink(destination=str(dest), overwrite=False)
        sink.prepare_artifacts({
            "any": [Artifact(id="a1", type="file/text", path=str(src), metadata={})]
        })

        with pytest.raises(FileExistsError, match="exists"):
            sink.write({"results": []})

    def test_metadata_overrides_security_level(self, tmp_path):
        """Metadata security_level overrides source artifact level."""
        src = tmp_path / "source.txt"
        src.write_text("data")

        sink = FileCopySink(destination=str(tmp_path / "dest.txt"))
        sink.prepare_artifacts({
            "any": [Artifact(
                id="a1",
                type="file/text",
                path=str(src),
                metadata={},
                security_level="official",
            )]
        })

        sink.write({"results": []}, metadata={"security_level": "SECRET"})

        assert sink._security_level == "secret"


class TestFileCopySinkArtifacts:
    """Tests for artifact collection."""

    def test_collect_artifacts_returns_copied_file(self, tmp_path):
        """Collect artifacts returns copied file info."""
        src = tmp_path / "source.txt"
        src.write_text("hello")

        dest = tmp_path / "dest.txt"
        sink = FileCopySink(destination=str(dest))
        sink.prepare_artifacts({
            "any": [Artifact(
                id="a1",
                type="file/text",
                path=str(src),
                metadata={"content_type": "text/plain"},
            )]
        })

        sink.write({"results": []}, metadata={"security_level": "official"})
        artifacts = sink.collect_artifacts()

        assert "file" in artifacts
        artifact = artifacts["file"]
        assert artifact.path == str(dest)
        assert artifact.type == "file/text"
        assert artifact.metadata["source"] == "a1"
        assert artifact.metadata["source_path"] == str(src)
        assert artifact.metadata["security_level"] == "official"

    def test_preserves_source_content_type(self, tmp_path):
        """Preserves content_type from source artifact."""
        src = tmp_path / "source.json"
        src.write_text('{}')

        sink = FileCopySink(destination=str(tmp_path / "dest.json"))
        sink.prepare_artifacts({
            "any": [Artifact(
                id="a1",
                type="file/json",
                path=str(src),
                metadata={"content_type": "application/json"},
            )]
        })

        sink.write({"results": []})
        artifacts = sink.collect_artifacts()

        assert artifacts["file"].metadata["content_type"] == "application/json"

    def test_collect_artifacts_clears_state(self, tmp_path):
        """Collect artifacts clears internal state."""
        src = tmp_path / "source.txt"
        src.write_text("hello")

        sink = FileCopySink(destination=str(tmp_path / "dest.txt"))
        sink.prepare_artifacts({
            "any": [Artifact(id="a1", type="file/text", path=str(src), metadata={})]
        })

        sink.write({"results": []})

        # First collection
        artifacts = sink.collect_artifacts()
        assert "file" in artifacts

        # Second collection should be empty
        artifacts2 = sink.collect_artifacts()
        assert artifacts2 == {}

    def test_collect_artifacts_empty_before_write(self, tmp_path):
        """Collect artifacts returns empty dict before write."""
        sink = FileCopySink(destination=str(tmp_path / "dest.txt"))

        artifacts = sink.collect_artifacts()
        assert artifacts == {}

    def test_defaults_to_octet_stream_type(self, tmp_path):
        """Defaults to octet-stream when no type available."""
        src = tmp_path / "source.bin"
        src.write_bytes(b"binary data")

        sink = FileCopySink(destination=str(tmp_path / "dest.bin"))
        # No type specified in artifact - simulated by setting _output_type to None
        sink.prepare_artifacts({
            "any": [Artifact(id="a1", type="", path=str(src), metadata={})]
        })
        sink._output_type = None  # Force no type

        sink.write({"results": []})
        artifacts = sink.collect_artifacts()

        assert artifacts["file"].type == "file/octet-stream"


class TestFileCopyBinaryFiles:
    """Tests for binary file handling."""

    def test_copies_binary_files(self, tmp_path):
        """Correctly copies binary files."""
        src = tmp_path / "source.bin"
        binary_data = bytes(range(256))
        src.write_bytes(binary_data)

        dest = tmp_path / "dest.bin"
        sink = FileCopySink(destination=str(dest))
        sink.prepare_artifacts({
            "any": [Artifact(id="a1", type="file/binary", path=str(src), metadata={})]
        })

        sink.write({"results": []})

        assert dest.read_bytes() == binary_data

    def test_copies_large_files(self, tmp_path):
        """Handles large files correctly."""
        src = tmp_path / "large.bin"
        large_data = b"x" * (10 * 1024 * 1024)  # 10MB
        src.write_bytes(large_data)

        dest = tmp_path / "dest.bin"
        sink = FileCopySink(destination=str(dest))
        sink.prepare_artifacts({
            "any": [Artifact(id="a1", type="file/binary", path=str(src), metadata={})]
        })

        sink.write({"results": []})

        assert dest.stat().st_size == src.stat().st_size
