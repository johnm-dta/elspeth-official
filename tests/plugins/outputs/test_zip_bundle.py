"""Tests for ZipResultSink."""

import json
import zipfile
from io import BytesIO

import pytest

from elspeth.core.interfaces import Artifact
from elspeth.plugins.outputs.zip_bundle import ZipResultSink


class TestZipResultSinkBasic:
    """Basic functionality tests."""

    def test_writes_zip_with_manifest_and_results(self, tmp_path):
        """Creates ZIP with manifest.json and results.json."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="bundle",
            timestamped=False,
            include_manifest=True,
            include_results=True,
        )

        payload = {
            "results": [
                {"row": {"text": "hello"}, "response": {"content": "hi"}},
            ],
        }

        sink.write(payload, metadata={"experiment": "test"})

        zip_path = tmp_path / "bundle.zip"
        assert zip_path.exists()

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "results.json" in names

            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["rows"] == 1
            assert manifest["metadata"]["experiment"] == "test"

            results = json.loads(zf.read("results.json"))
            assert len(results["results"]) == 1

    def test_timestamped_filename(self, tmp_path):
        """Timestamped bundles include timestamp in filename."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="test",
            timestamped=True,
        )

        sink.write({"results": []}, metadata={})

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name.startswith("test_")
        assert "T" in files[0].name  # ISO timestamp format
        assert files[0].suffix == ".zip"

    def test_bundle_name_from_metadata(self, tmp_path):
        """Bundle name derived from metadata when not specified."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name=None,
            timestamped=False,
        )

        sink.write({"results": []}, metadata={"experiment": "my_experiment"})

        assert (tmp_path / "my_experiment.zip").exists()


class TestZipResultSinkCSV:
    """Tests for CSV inclusion."""

    def test_include_csv_renders_results(self, tmp_path):
        """CSV includes row data and LLM content."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="with_csv",
            timestamped=False,
            include_csv=True,
        )

        payload = {
            "results": [
                {"row": {"id": 1, "text": "hello"}, "response": {"content": "response1"}},
                {"row": {"id": 2, "text": "world"}, "response": {"content": "response2"}},
            ],
        }

        sink.write(payload, metadata={})

        with zipfile.ZipFile(tmp_path / "with_csv.zip", "r") as zf:
            assert "results.csv" in zf.namelist()
            csv_content = zf.read("results.csv").decode("utf-8")
            assert "id" in csv_content
            assert "text" in csv_content
            assert "llm_content" in csv_content
            assert "response1" in csv_content
            assert "response2" in csv_content

    def test_csv_with_multiple_responses(self, tmp_path):
        """CSV handles multiple named responses."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="multi_resp",
            timestamped=False,
            include_csv=True,
        )

        payload = {
            "results": [
                {
                    "row": {"id": 1},
                    "responses": {
                        "baseline": {"content": "baseline_answer"},
                        "variant": {"content": "variant_answer"},
                    },
                },
            ],
        }

        sink.write(payload, metadata={})

        with zipfile.ZipFile(tmp_path / "multi_resp.zip", "r") as zf:
            csv_content = zf.read("results.csv").decode("utf-8")
            assert "llm_baseline" in csv_content
            assert "llm_variant" in csv_content
            assert "baseline_answer" in csv_content
            assert "variant_answer" in csv_content

    def test_csv_empty_results(self, tmp_path):
        """CSV is empty string when no results."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="empty",
            timestamped=False,
            include_csv=True,
        )

        sink.write({"results": []}, metadata={})

        with zipfile.ZipFile(tmp_path / "empty.zip", "r") as zf:
            csv_content = zf.read("results.csv").decode("utf-8")
            assert csv_content == ""


class TestZipResultSinkManifest:
    """Tests for manifest content."""

    def test_manifest_includes_aggregates(self, tmp_path):
        """Manifest includes aggregates when present."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="agg",
            timestamped=False,
        )

        payload = {
            "results": [{"row": {}}],
            "aggregates": {"mean_score": 0.85, "count": 10},
        }

        sink.write(payload, metadata={})

        with zipfile.ZipFile(tmp_path / "agg.zip", "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["aggregates"] == {"mean_score": 0.85, "count": 10}

    def test_manifest_includes_cost_summary(self, tmp_path):
        """Manifest includes cost_summary when present."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="cost",
            timestamped=False,
        )

        payload = {
            "results": [],
            "cost_summary": {"total_tokens": 5000, "total_cost": 0.25},
        }

        sink.write(payload, metadata={})

        with zipfile.ZipFile(tmp_path / "cost.zip", "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["cost_summary"] == {"total_tokens": 5000, "total_cost": 0.25}

    def test_manifest_includes_failures(self, tmp_path):
        """Manifest includes failures when present."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="fail",
            timestamped=False,
        )

        payload = {
            "results": [],
            "failures": [{"row_id": 1, "error": "timeout"}],
        }

        sink.write(payload, metadata={})

        with zipfile.ZipFile(tmp_path / "fail.zip", "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["failures"] == [{"row_id": 1, "error": "timeout"}]

    def test_manifest_generated_at_timestamp(self, tmp_path):
        """Manifest includes generated_at ISO timestamp."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="ts",
            timestamped=False,
        )

        sink.write({"results": []}, metadata={})

        with zipfile.ZipFile(tmp_path / "ts.zip", "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert "generated_at" in manifest
            assert "T" in manifest["generated_at"]  # ISO format


class TestZipResultSinkValidation:
    """Tests for configuration validation."""

    def test_invalid_on_error_raises(self, tmp_path):
        """Invalid on_error value raises ValueError."""
        with pytest.raises(ValueError, match="on_error must be 'abort'"):
            ZipResultSink(
                base_path=tmp_path,
                on_error="ignore",
            )

    def test_valid_on_error_abort(self, tmp_path):
        """on_error='abort' is accepted."""
        sink = ZipResultSink(
            base_path=tmp_path,
            on_error="abort",
        )
        assert sink.on_error == "abort"


class TestZipResultSinkOptions:
    """Tests for optional features."""

    def test_exclude_manifest(self, tmp_path):
        """Can exclude manifest from bundle."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="no_manifest",
            timestamped=False,
            include_manifest=False,
        )

        sink.write({"results": []}, metadata={})

        with zipfile.ZipFile(tmp_path / "no_manifest.zip", "r") as zf:
            assert "manifest.json" not in zf.namelist()
            assert "results.json" in zf.namelist()

    def test_exclude_results(self, tmp_path):
        """Can exclude results from bundle."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="no_results",
            timestamped=False,
            include_results=False,
        )

        sink.write({"results": []}, metadata={})

        with zipfile.ZipFile(tmp_path / "no_results.zip", "r") as zf:
            assert "results.json" not in zf.namelist()
            assert "manifest.json" in zf.namelist()

    def test_custom_filenames(self, tmp_path):
        """Custom manifest/results/csv filenames."""
        sink = ZipResultSink(
            base_path=tmp_path,
            bundle_name="custom",
            timestamped=False,
            include_csv=True,
            manifest_name="meta.json",
            results_name="output.json",
            csv_name="data.csv",
        )

        sink.write({"results": [{"row": {"x": 1}}]}, metadata={})

        with zipfile.ZipFile(tmp_path / "custom.zip", "r") as zf:
            names = zf.namelist()
            assert "meta.json" in names
            assert "output.json" in names
            assert "data.csv" in names


class TestZipResultSinkArtifactReading:
    """Tests for _read_artifact helper."""

    def test_read_artifact_from_path(self, tmp_path):
        """Reads artifact content from file path."""
        content = b"file content here"
        file_path = tmp_path / "test.txt"
        file_path.write_bytes(content)

        artifact = Artifact(id="test", type="file/text", path=str(file_path))
        result = ZipResultSink._read_artifact(artifact)

        assert result == content

    def test_read_artifact_from_bytes_payload(self):
        """Reads artifact from bytes payload."""
        content = b"raw bytes"
        artifact = Artifact(id="test", type="file/binary", payload=content)

        result = ZipResultSink._read_artifact(artifact)

        assert result == content

    def test_read_artifact_from_bytearray_payload(self):
        """Reads artifact from bytearray payload."""
        content = bytearray(b"byte array data")
        artifact = Artifact(id="test", type="file/binary", payload=content)

        result = ZipResultSink._read_artifact(artifact)

        assert result == b"byte array data"

    def test_read_artifact_from_readable_bytes(self):
        """Reads artifact from file-like object returning bytes."""
        content = b"readable bytes"
        buffer = BytesIO(content)
        artifact = Artifact(id="test", type="file/binary", payload=buffer)

        result = ZipResultSink._read_artifact(artifact)

        assert result == content

    def test_read_artifact_from_dict_payload(self):
        """Serializes dict payload to JSON bytes."""
        payload = {"key": "value", "number": 42}
        artifact = Artifact(id="test", type="application/json", payload=payload)

        result = ZipResultSink._read_artifact(artifact)

        assert json.loads(result) == payload

    def test_read_artifact_missing_data_raises(self):
        """Raises ValueError when artifact has no path or payload."""
        artifact = Artifact(id="test", type="file/text")

        with pytest.raises(ValueError, match="missing payload data"):
            ZipResultSink._read_artifact(artifact)
