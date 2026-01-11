"""Tests for Analytics Report Sink."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from elspeth.plugins.outputs.analytics_report import AnalyticsReportSink


class TestAnalyticsReportSinkInit:
    """Tests for AnalyticsReportSink initialization."""

    def test_init_with_defaults(self, tmp_path: Path):
        """Sink initializes with default values."""
        sink = AnalyticsReportSink(base_path=str(tmp_path))

        assert sink.base_path == tmp_path
        assert sink.file_stem == "analytics_report"
        assert sink.formats == ["json", "md"]
        assert sink.include_metadata is True
        assert sink.include_aggregates is True
        assert sink.include_comparisons is True

    def test_init_with_custom_options(self, tmp_path: Path):
        """Sink accepts custom options."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            file_stem="custom_report",
            formats=["json"],
            include_metadata=False,
            include_aggregates=False,
            include_comparisons=False,
        )

        assert sink.file_stem == "custom_report"
        assert sink.formats == ["json"]
        assert sink.include_metadata is False
        assert sink.include_aggregates is False
        assert sink.include_comparisons is False

    def test_init_normalizes_markdown_format(self, tmp_path: Path):
        """Format 'markdown' is normalized to 'md'."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["markdown"],
        )

        assert sink.formats == ["md"]

    def test_init_filters_invalid_formats(self, tmp_path: Path):
        """Invalid formats are filtered out."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json", "invalid", "md", "csv"],
        )

        assert sink.formats == ["json", "md"]

    def test_init_defaults_to_json_md_for_empty_formats(self, tmp_path: Path):
        """Empty formats list defaults to json and md."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=[],
        )

        # Empty list is falsy, so falls back to default ["json", "md"]
        assert sink.formats == ["json", "md"]

    def test_init_invalid_on_error(self, tmp_path: Path):
        """Rejects invalid on_error value."""
        with pytest.raises(ValueError, match="on_error must be 'abort'"):
            AnalyticsReportSink(base_path=str(tmp_path), on_error="ignore")


class TestWriteJsonFormat:
    """Tests for JSON output generation."""

    def test_write_creates_json_file(self, tmp_path: Path):
        """JSON file is created with correct content."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
        )

        results = {
            "results": [{"id": 1}, {"id": 2}],
            "failures": [],
        }
        sink.write(results, metadata={})

        json_path = tmp_path / "analytics_report.json"
        assert json_path.exists()

        content = json.loads(json_path.read_text())
        assert content["rows"] == 2
        assert content["failures"] == 0

    def test_write_includes_failure_count(self, tmp_path: Path):
        """Failure count is included in output."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
        )

        results = {
            "results": [{"id": 1}],
            "failures": [{"error": "test"}, {"error": "test2"}],
        }
        sink.write(results, metadata={})

        json_path = tmp_path / "analytics_report.json"
        content = json.loads(json_path.read_text())

        assert content["failures"] == 2
        assert len(content["failure_examples"]) == 2

    def test_write_limits_failure_examples(self, tmp_path: Path):
        """Only first 5 failures are included as examples."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
        )

        results = {
            "results": [],
            "failures": [{"error": f"error_{i}"} for i in range(10)],
        }
        sink.write(results, metadata={})

        json_path = tmp_path / "analytics_report.json"
        content = json.loads(json_path.read_text())

        assert content["failures"] == 10
        assert len(content["failure_examples"]) == 5

    def test_write_includes_aggregates(self, tmp_path: Path):
        """Aggregates are included when enabled."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
            include_aggregates=True,
        )

        results = {
            "results": [],
            "aggregates": {"mean_score": 0.85, "count": 100},
        }
        sink.write(results, metadata={})

        json_path = tmp_path / "analytics_report.json"
        content = json.loads(json_path.read_text())

        assert content["aggregates"] == {"mean_score": 0.85, "count": 100}

    def test_write_excludes_aggregates_when_disabled(self, tmp_path: Path):
        """Aggregates are excluded when disabled."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
            include_aggregates=False,
        )

        results = {
            "results": [],
            "aggregates": {"mean_score": 0.85},
        }
        sink.write(results, metadata={})

        json_path = tmp_path / "analytics_report.json"
        content = json.loads(json_path.read_text())

        assert "aggregates" not in content

    def test_write_includes_baseline_comparison(self, tmp_path: Path):
        """Baseline comparison is included when enabled."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
            include_comparisons=True,
        )

        results = {
            "results": [],
            "baseline_comparison": {"delta": 0.05, "significant": True},
        }
        sink.write(results, metadata={})

        json_path = tmp_path / "analytics_report.json"
        content = json.loads(json_path.read_text())

        assert content["baseline_comparison"] == {"delta": 0.05, "significant": True}

    def test_write_includes_analytics_sections(self, tmp_path: Path):
        """Analytics sections are included in output."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
        )

        results = {
            "results": [],
            "score_cliffs_delta": {"delta": 0.3, "interpretation": "medium"},
            "score_assumptions": {"normality_p": 0.12},
            "score_practical": {"nnt": 5.2},
        }
        sink.write(results, metadata={})

        json_path = tmp_path / "analytics_report.json"
        content = json.loads(json_path.read_text())

        assert "analytics" in content
        assert content["analytics"]["score_cliffs_delta"] == {"delta": 0.3, "interpretation": "medium"}
        assert content["analytics"]["score_assumptions"] == {"normality_p": 0.12}
        assert content["analytics"]["score_practical"] == {"nnt": 5.2}


class TestWriteMarkdownFormat:
    """Tests for Markdown output generation."""

    def test_write_creates_markdown_file(self, tmp_path: Path):
        """Markdown file is created."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["md"],
        )

        results = {"results": [{"id": 1}], "failures": []}
        sink.write(results, metadata={})

        md_path = tmp_path / "analytics_report.md"
        assert md_path.exists()

    def test_markdown_includes_header(self, tmp_path: Path):
        """Markdown includes report header."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["md"],
        )

        results = {"results": []}
        sink.write(results, metadata={})

        md_path = tmp_path / "analytics_report.md"
        content = md_path.read_text()

        assert "# Analytics Report" in content

    def test_markdown_includes_row_count(self, tmp_path: Path):
        """Markdown includes row count."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["md"],
        )

        results = {"results": [{"id": i} for i in range(42)]}
        sink.write(results, metadata={})

        md_path = tmp_path / "analytics_report.md"
        content = md_path.read_text()

        assert "**Rows processed:** 42" in content

    def test_markdown_includes_failure_count(self, tmp_path: Path):
        """Markdown includes failure count when failures exist."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["md"],
        )

        results = {"results": [], "failures": [{"error": "test"}]}
        sink.write(results, metadata={})

        md_path = tmp_path / "analytics_report.md"
        content = md_path.read_text()

        assert "**Failures:** 1" in content

    def test_markdown_includes_aggregates_section(self, tmp_path: Path):
        """Markdown includes aggregates section."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["md"],
            include_aggregates=True,
        )

        results = {
            "results": [],
            "aggregates": {"mean": 0.8, "std": 0.1},
        }
        sink.write(results, metadata={})

        md_path = tmp_path / "analytics_report.md"
        content = md_path.read_text()

        assert "## Aggregates" in content
        assert '"mean": 0.8' in content

    def test_markdown_includes_baseline_comparison_section(self, tmp_path: Path):
        """Markdown includes baseline comparison section."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["md"],
            include_comparisons=True,
        )

        results = {
            "results": [],
            "baseline_comparison": {"delta": 0.05},
        }
        sink.write(results, metadata={})

        md_path = tmp_path / "analytics_report.md"
        content = md_path.read_text()

        assert "## Baseline Comparison" in content
        assert '"delta": 0.05' in content

    def test_markdown_includes_analytics_sections(self, tmp_path: Path):
        """Markdown includes individual analytics sections."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["md"],
        )

        results = {
            "results": [],
            "score_cliffs_delta": {"delta": 0.3},
            "score_significance": {"p_value": 0.001},
        }
        sink.write(results, metadata={})

        md_path = tmp_path / "analytics_report.md"
        content = md_path.read_text()

        assert "## Analytics" in content
        assert "### score_cliffs_delta" in content
        assert "### score_significance" in content

    def test_markdown_includes_failure_examples_section(self, tmp_path: Path):
        """Markdown includes failure examples section."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["md"],
        )

        results = {
            "results": [],
            "failures": [{"error": "Error 1"}, {"error": "Error 2"}],
        }
        sink.write(results, metadata={})

        md_path = tmp_path / "analytics_report.md"
        content = md_path.read_text()

        assert "## Failure Examples" in content
        assert "Error 1" in content

    def test_markdown_includes_metadata_fields(self, tmp_path: Path):
        """Markdown includes metadata fields when present."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["md"],
            include_metadata=True,
        )

        results = {
            "results": [],
            "metadata": {
                "early_stop": {"triggered": True, "reason": "threshold"},
                "cost_summary": {"total_cost": 1.50},
            },
        }
        sink.write(results, metadata={})

        md_path = tmp_path / "analytics_report.md"
        content = md_path.read_text()

        assert "Early stop" in content
        assert "Cost summary" in content


class TestBothFormats:
    """Tests for writing both JSON and Markdown."""

    def test_write_creates_both_files(self, tmp_path: Path):
        """Both JSON and Markdown files are created."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json", "md"],
        )

        results = {"results": []}
        sink.write(results, metadata={})

        assert (tmp_path / "analytics_report.json").exists()
        assert (tmp_path / "analytics_report.md").exists()

    def test_write_with_custom_file_stem(self, tmp_path: Path):
        """Custom file stem is used for both files."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            file_stem="my_custom_report",
            formats=["json", "md"],
        )

        results = {"results": []}
        sink.write(results, metadata={})

        assert (tmp_path / "my_custom_report.json").exists()
        assert (tmp_path / "my_custom_report.md").exists()


class TestMetadataHandling:
    """Tests for metadata handling."""

    def test_metadata_includes_retry_summary(self, tmp_path: Path):
        """Retry summary is included in metadata."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
            include_metadata=True,
        )

        results = {
            "results": [],
            "metadata": {
                "retry_summary": {"total_retries": 5, "success_rate": 0.95},
            },
        }
        sink.write(results, metadata={})

        json_path = tmp_path / "analytics_report.json"
        content = json.loads(json_path.read_text())

        assert content["metadata"]["retry_summary"] == {"total_retries": 5, "success_rate": 0.95}

    def test_metadata_includes_security_level(self, tmp_path: Path):
        """Security level is included in metadata."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
            include_metadata=True,
        )

        results = {
            "results": [],
            "metadata": {
                "security_level": "official-sensitive",
            },
        }
        sink.write(results, metadata={})

        json_path = tmp_path / "analytics_report.json"
        content = json.loads(json_path.read_text())

        assert content["metadata"]["security_level"] == "official-sensitive"

    def test_metadata_excluded_when_disabled(self, tmp_path: Path):
        """Metadata section is excluded when disabled."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
            include_metadata=False,
        )

        results = {
            "results": [],
            "metadata": {"retry_summary": {"total_retries": 5}},
        }
        sink.write(results, metadata={})

        json_path = tmp_path / "analytics_report.json"
        content = json.loads(json_path.read_text())

        assert "metadata" not in content


class TestArtifactCollection:
    """Tests for artifact collection."""

    def test_collect_artifacts_returns_written_files(self, tmp_path: Path):
        """Collect artifacts returns the files that were written."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json", "md"],
        )

        sink.write({"results": []}, metadata={})
        artifacts = sink.collect_artifacts()

        assert len(artifacts) == 2
        assert "analytics_report.json" in artifacts
        assert "analytics_report.md" in artifacts

    def test_collect_artifacts_with_security_level(self, tmp_path: Path):
        """Artifacts include security level from metadata."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
        )

        sink.write({"results": []}, metadata={"security_level": "official"})
        artifacts = sink.collect_artifacts()

        assert artifacts["analytics_report.json"].security_level == "official"

    def test_collect_artifacts_clears_internal_list(self, tmp_path: Path):
        """Collecting artifacts clears the internal list."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
        )

        sink.write({"results": []}, metadata={})

        # First collection returns artifacts
        artifacts = sink.collect_artifacts()
        assert len(artifacts) == 1

        # Second collection returns empty
        artifacts = sink.collect_artifacts()
        assert len(artifacts) == 0

    def test_artifact_content_types(self, tmp_path: Path):
        """Artifacts have correct content types."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json", "md"],
        )

        sink.write({"results": []}, metadata={})
        artifacts = sink.collect_artifacts()

        assert artifacts["analytics_report.json"].type == "application/json"
        assert artifacts["analytics_report.md"].type == "text/markdown"


class TestEmptyAndMinimalInputs:
    """Tests for handling empty and minimal inputs."""

    def test_write_with_empty_results(self, tmp_path: Path):
        """Handles empty results list."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
        )

        sink.write({"results": []}, metadata={})

        json_path = tmp_path / "analytics_report.json"
        content = json.loads(json_path.read_text())

        assert content["rows"] == 0

    def test_write_with_missing_results_key(self, tmp_path: Path):
        """Handles missing results key."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
        )

        sink.write({}, metadata={})

        json_path = tmp_path / "analytics_report.json"
        content = json.loads(json_path.read_text())

        assert content["rows"] == 0

    def test_write_with_none_metadata(self, tmp_path: Path):
        """Handles None metadata."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json"],
        )

        sink.write({"results": []}, metadata=None)

        json_path = tmp_path / "analytics_report.json"
        assert json_path.exists()

    def test_write_with_no_optional_sections(self, tmp_path: Path):
        """Handles payload with no optional sections."""
        sink = AnalyticsReportSink(
            base_path=str(tmp_path),
            formats=["json", "md"],
        )

        results = {"results": [{"id": 1}]}
        sink.write(results, metadata={})

        # Both files should be created
        assert (tmp_path / "analytics_report.json").exists()
        assert (tmp_path / "analytics_report.md").exists()


class TestDirectoryCreation:
    """Tests for directory creation behavior."""

    def test_write_creates_base_directory(self, tmp_path: Path):
        """Base directory is created if it doesn't exist."""
        nested_path = tmp_path / "deeply" / "nested" / "directory"

        sink = AnalyticsReportSink(
            base_path=str(nested_path),
            formats=["json"],
        )

        sink.write({"results": []}, metadata={})

        assert nested_path.exists()
        assert (nested_path / "analytics_report.json").exists()


class TestProducesConsumes:
    """Tests for produces/consumes interface."""

    def test_produces_returns_descriptor(self, tmp_path: Path):
        """Produces returns artifact descriptor."""
        sink = AnalyticsReportSink(base_path=str(tmp_path))

        descriptors = sink.produces()

        assert len(descriptors) == 1
        assert descriptors[0].name == "analytics_report"
        assert descriptors[0].type == "application/json"

    def test_consumes_returns_empty(self, tmp_path: Path):
        """Consumes returns empty list."""
        sink = AnalyticsReportSink(base_path=str(tmp_path))

        consumed = sink.consumes()

        assert consumed == []
