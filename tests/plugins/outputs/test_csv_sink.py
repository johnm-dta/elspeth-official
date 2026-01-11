"""Tests for CSV result sink."""

from __future__ import annotations

import pandas as pd
import pytest

from elspeth.plugins.outputs.csv_file import CsvResultSink


class TestCsvResultSinkInit:
    """Tests for CsvResultSink initialization."""

    def test_init_with_defaults(self, tmp_path):
        """Initializes with default values."""
        sink = CsvResultSink(path=str(tmp_path / "out.csv"))

        assert sink.path == tmp_path / "out.csv"
        assert sink.overwrite is True
        assert sink.on_error == "abort"

    def test_init_with_custom_options(self, tmp_path):
        """Accepts custom options."""
        sink = CsvResultSink(
            path=str(tmp_path / "custom.csv"),
            overwrite=False,
        )

        assert sink.path == tmp_path / "custom.csv"
        assert sink.overwrite is False

    def test_init_rejects_invalid_on_error(self, tmp_path):
        """Rejects invalid on_error value."""
        with pytest.raises(ValueError, match="on_error must be 'abort'"):
            CsvResultSink(path=str(tmp_path / "out.csv"), on_error="ignore")


class TestCsvResultSinkWrite:
    """Tests for CsvResultSink.write()."""

    def test_write_creates_csv_file(self, tmp_path):
        """Write creates CSV file."""
        sink_path = tmp_path / "out.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {
            "results": [
                {"row": {"text": "hello"}, "response": {"content": "hi"}},
            ]
        }
        sink.write(payload)

        assert sink_path.exists()

    def test_write_includes_row_data(self, tmp_path):
        """Write includes row data in CSV."""
        sink_path = tmp_path / "out.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {
            "results": [
                {"row": {"id": 1, "text": "hello"}, "response": {"content": "hi"}},
                {"row": {"id": 2, "text": "world"}, "response": {"content": "hey"}},
            ]
        }
        sink.write(payload)

        df = pd.read_csv(sink_path)
        assert list(df["id"]) == [1, 2]
        assert list(df["text"]) == ["hello", "world"]

    def test_write_includes_llm_content(self, tmp_path):
        """Write includes LLM content column."""
        sink_path = tmp_path / "out.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {
            "results": [
                {"row": {"text": "hello"}, "response": {"content": "response text"}},
            ]
        }
        sink.write(payload)

        df = pd.read_csv(sink_path)
        assert df["llm_content"].iloc[0] == "response text"

    def test_write_includes_multiple_responses(self, tmp_path):
        """Write includes multiple response columns."""
        sink_path = tmp_path / "out.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {
            "results": [
                {
                    "row": {"text": "test"},
                    "response": {"content": "main"},
                    "responses": {
                        "variant_a": {"content": "response a"},
                        "variant_b": {"content": "response b"},
                    },
                },
            ]
        }
        sink.write(payload)

        df = pd.read_csv(sink_path)
        assert df["llm_content"].iloc[0] == "main"
        assert df["llm_variant_a"].iloc[0] == "response a"
        assert df["llm_variant_b"].iloc[0] == "response b"

    def test_write_creates_parent_directories(self, tmp_path):
        """Write creates parent directories if needed."""
        sink_path = tmp_path / "nested" / "path" / "out.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {"results": [{"row": {"x": 1}, "response": {"content": "y"}}]}
        sink.write(payload)

        assert sink_path.exists()

    def test_write_empty_results(self, tmp_path):
        """Write handles empty results."""
        sink_path = tmp_path / "empty.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {"results": []}
        sink.write(payload)

        # Empty DataFrame writes an empty file (no columns/rows)
        assert sink_path.exists()
        content = sink_path.read_text()
        # pandas writes an empty string for empty DataFrame
        assert content == "" or content == "\n"

    def test_write_no_results_key(self, tmp_path):
        """Write handles missing results key."""
        sink_path = tmp_path / "missing.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {}
        sink.write(payload)

        # Empty DataFrame writes an empty file
        assert sink_path.exists()
        content = sink_path.read_text()
        assert content == "" or content == "\n"


class TestCsvResultSinkOverwrite:
    """Tests for overwrite behavior."""

    def test_overwrites_existing_file_when_enabled(self, tmp_path):
        """Overwrites existing file when overwrite=True."""
        sink_path = tmp_path / "out.csv"
        sink_path.write_text("old content")

        sink = CsvResultSink(path=str(sink_path), overwrite=True)
        payload = {"results": [{"row": {"x": 1}, "response": {"content": "new"}}]}
        sink.write(payload)

        content = sink_path.read_text()
        assert "old content" not in content
        assert "1" in content

    def test_raises_when_exists_and_no_overwrite(self, tmp_path):
        """Raises FileExistsError when file exists and overwrite=False."""
        sink_path = tmp_path / "out.csv"
        sink_path.write_text("existing")

        sink = CsvResultSink(path=str(sink_path), overwrite=False)
        payload = {"results": [{"row": {"x": 1}, "response": {"content": "y"}}]}

        with pytest.raises(FileExistsError, match="exists"):
            sink.write(payload)


class TestCsvResultSinkSpecialCharacters:
    """Tests for special character handling."""

    def test_handles_unicode_content(self, tmp_path):
        """Handles unicode characters correctly."""
        sink_path = tmp_path / "unicode.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {
            "results": [
                {"row": {"text": "Hello 世界"}, "response": {"content": "Привет мир"}},
            ]
        }
        sink.write(payload)

        df = pd.read_csv(sink_path)
        assert df["text"].iloc[0] == "Hello 世界"
        assert df["llm_content"].iloc[0] == "Привет мир"

    def test_handles_commas_in_content(self, tmp_path):
        """Handles commas in content correctly."""
        sink_path = tmp_path / "commas.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {
            "results": [
                {"row": {"text": "a, b, c"}, "response": {"content": "one, two, three"}},
            ]
        }
        sink.write(payload)

        df = pd.read_csv(sink_path)
        assert df["text"].iloc[0] == "a, b, c"
        assert df["llm_content"].iloc[0] == "one, two, three"

    def test_handles_newlines_in_content(self, tmp_path):
        """Handles newlines in content correctly."""
        sink_path = tmp_path / "newlines.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {
            "results": [
                {"row": {"text": "line1\nline2"}, "response": {"content": "a\nb\nc"}},
            ]
        }
        sink.write(payload)

        df = pd.read_csv(sink_path)
        assert df["text"].iloc[0] == "line1\nline2"

    def test_handles_quotes_in_content(self, tmp_path):
        """Handles quotes in content correctly."""
        sink_path = tmp_path / "quotes.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {
            "results": [
                {"row": {"text": 'He said "hello"'}, "response": {"content": "response"}},
            ]
        }
        sink.write(payload)

        df = pd.read_csv(sink_path)
        assert df["text"].iloc[0] == 'He said "hello"'


class TestCsvResultSinkArtifacts:
    """Tests for artifact collection."""

    def test_collect_artifacts_after_write(self, tmp_path):
        """Collect artifacts returns CSV path."""
        sink_path = tmp_path / "out.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {"results": [{"row": {"x": 1}, "response": {"content": "y"}}]}
        sink.write(payload, metadata={"security_level": "official"})
        artifacts = sink.collect_artifacts()

        assert "csv" in artifacts
        artifact = artifacts["csv"]
        assert artifact.path == str(sink_path)
        assert artifact.type == "file/csv"
        assert artifact.metadata["security_level"] == "official"

    def test_collect_artifacts_clears_state(self, tmp_path):
        """Collect artifacts clears internal state."""
        sink_path = tmp_path / "out.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {"results": [{"row": {"x": 1}, "response": {"content": "y"}}]}
        sink.write(payload)

        # First collection
        artifacts = sink.collect_artifacts()
        assert "csv" in artifacts

        # Second collection should be empty
        artifacts2 = sink.collect_artifacts()
        assert artifacts2 == {}

    def test_collect_artifacts_empty_before_write(self, tmp_path):
        """Collect artifacts returns empty dict before write."""
        sink = CsvResultSink(path=str(tmp_path / "out.csv"))

        artifacts = sink.collect_artifacts()
        assert artifacts == {}

    def test_security_level_normalized(self, tmp_path):
        """Security level is normalized."""
        sink_path = tmp_path / "out.csv"
        sink = CsvResultSink(path=str(sink_path))

        payload = {"results": [{"row": {"x": 1}, "response": {"content": "y"}}]}
        sink.write(payload, metadata={"security_level": "OFFICIAL-SENSITIVE"})
        artifacts = sink.collect_artifacts()

        assert artifacts["csv"].security_level == "official-sensitive"


class TestCsvResultSinkProducesConsumes:
    """Tests for artifact descriptors."""

    def test_produces_returns_csv_descriptor(self, tmp_path):
        """Produces returns CSV descriptor."""
        sink = CsvResultSink(path=str(tmp_path / "out.csv"))

        descriptors = sink.produces()

        assert len(descriptors) == 1
        assert descriptors[0].name == "csv"
        assert descriptors[0].type == "file/csv"
        assert descriptors[0].persist is True

    def test_consumes_returns_empty(self, tmp_path):
        """Consumes returns empty list."""
        sink = CsvResultSink(path=str(tmp_path / "out.csv"))

        assert sink.consumes() == []

    def test_finalize_returns_none(self, tmp_path):
        """Finalize returns None."""
        sink = CsvResultSink(path=str(tmp_path / "out.csv"))

        result = sink.finalize({})
        assert result is None
