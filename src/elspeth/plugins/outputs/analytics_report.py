"""Result sink that emits structured analytics summaries."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elspeth.core.interfaces import Artifact, ArtifactDescriptor, ResultSink
from elspeth.core.security import normalize_security_level

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class AnalyticsReportSink(ResultSink):
    """Write experiment analytics to JSON/Markdown artifacts."""

    def __init__(
        self,
        *,
        base_path: str,
        file_stem: str = "analytics_report",
        formats: Sequence[str] | None = None,
        include_metadata: bool = True,
        include_aggregates: bool = True,
        include_comparisons: bool = True,
        on_error: str = "abort",
    ) -> None:
        self.base_path = Path(base_path)
        self.file_stem = file_stem or "analytics_report"
        selected = []
        for fmt in formats or ["json", "md"]:
            normalized = (fmt or "").strip().lower()
            if normalized == "markdown":
                normalized = "md"
            if normalized:
                selected.append(normalized)
        self.formats = [fmt for fmt in selected if fmt in {"json", "md"}] or ["json"]
        if on_error != "abort":
            raise ValueError("on_error must be 'abort'")
        self.on_error = on_error
        self.include_metadata = include_metadata
        self.include_aggregates = include_aggregates
        self.include_comparisons = include_comparisons
        self._last_written_files: list[Path] = []
        self._security_level: str | None = None

    def write(self, results: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        try:
            summary = self._build_summary(results, metadata or {})
            self.base_path.mkdir(parents=True, exist_ok=True)
            written: list[Path] = []
            if "json" in self.formats:
                path = self.base_path / f"{self.file_stem}.json"
                path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
                written.append(path)
            if "md" in self.formats:
                path = self.base_path / f"{self.file_stem}.md"
                path.write_text(self._render_markdown(summary), encoding="utf-8")
                written.append(path)
            self._last_written_files = written
            if metadata:
                self._security_level = normalize_security_level(metadata.get("security_level"))
        except Exception:  # pragma: no cover - error handling path
            raise

    def produces(self) -> list[ArtifactDescriptor]:  # pragma: no cover - metadata only
        return [
            ArtifactDescriptor(name="analytics_report", type="application/json", persist=True, alias="analytics"),
        ]

    def consumes(self) -> list[str]:  # pragma: no cover - no dependencies
        return []

    def collect_artifacts(self) -> dict[str, Artifact]:
        artifacts: dict[str, Artifact] = {}
        for path in self._last_written_files:
            content_type = "application/json" if path.suffix == ".json" else "text/markdown"
            artifacts[path.name] = Artifact(
                id="",
                type=content_type,
                path=str(path),
                metadata={"path": str(path), "content_type": content_type},
                persist=True,
                security_level=self._security_level,
            )
        self._last_written_files = []
        return artifacts

    def _build_summary(self, payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
        results = list(payload.get("results") or [])
        failures = list(payload.get("failures") or [])
        summary: dict[str, Any] = {
            "rows": len(results),
            "failures": len(failures),
        }
        if failures:
            summary["failure_examples"] = failures[:5]
        payload_meta = payload.get("metadata") or {}
        if self.include_metadata:
            summary["metadata"] = {
                "retry_summary": payload_meta.get("retry_summary"),
                "early_stop": payload_meta.get("early_stop"),
                "cost_summary": payload_meta.get("cost_summary"),
                "security_level": payload_meta.get("security_level"),
            }
        if self.include_aggregates and payload.get("aggregates"):
            summary["aggregates"] = payload["aggregates"]
        if self.include_comparisons:
            summary["baseline_comparison"] = payload.get("baseline_comparison")
        analytics_sections = {}
        for key in (
            "score_cliffs_delta",
            "score_assumptions",
            "score_practical",
            "score_distribution",
            "score_significance",
            "score_power",
        ):
            value = payload.get(key)
            if value:
                analytics_sections[key] = value
        if analytics_sections:
            summary["analytics"] = analytics_sections
        return summary

    def _render_markdown(self, summary: Mapping[str, Any]) -> str:
        lines: list[str] = ["# Analytics Report", ""]
        lines.append(f"- **Rows processed:** {summary.get('rows', 0)}")
        failures = summary.get("failures", 0)
        if failures:
            lines.append(f"- **Failures:** {failures}")
        metadata = summary.get("metadata") or {}
        if metadata.get("early_stop"):
            lines.append(f"- **Early stop:** {json.dumps(metadata['early_stop'])}")
        if metadata.get("cost_summary"):
            lines.append(f"- **Cost summary:** {json.dumps(metadata['cost_summary'])}")
        lines.append("")

        aggregates = summary.get("aggregates")
        if aggregates:
            lines.extend(["## Aggregates", "```json", json.dumps(aggregates, indent=2, sort_keys=True), "```", ""])

        baseline_comparison = summary.get("baseline_comparison")
        if baseline_comparison:
            lines.extend(
                [
                    "## Baseline Comparison",
                    "```json",
                    json.dumps(baseline_comparison, indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )

        analytics = summary.get("analytics") or {}
        if analytics:
            lines.append("## Analytics")
            for name, value in analytics.items():
                lines.extend(
                    [
                        f"### {name}",
                        "```json",
                        json.dumps(value, indent=2, sort_keys=True),
                        "```",
                        "",
                    ]
                )

        if summary.get("failure_examples"):
            lines.extend(
                [
                    "## Failure Examples",
                    "```json",
                    json.dumps(summary["failure_examples"], indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        return "\n".join(lines)


# --- Plugin Registration ---
from elspeth.core.registry import ARTIFACTS_SECTION_SCHEMA, ON_ERROR_ENUM, registry

ANALYTICS_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "base_path": {"type": "string"},
        "file_stem": {"type": "string"},
        "formats": {
            "type": "array",
            "items": {"type": "string", "enum": ["json", "md", "markdown"]},
        },
        "include_metadata": {"type": "boolean"},
        "include_aggregates": {"type": "boolean"},
        "include_comparisons": {"type": "boolean"},
        "artifacts": ARTIFACTS_SECTION_SCHEMA,
        "security_level": {"type": "string"},
        "on_error": ON_ERROR_ENUM,
    },
    "required": ["base_path"],
    "additionalProperties": True,
}

registry.register_sink("analytics_report", AnalyticsReportSink, ANALYTICS_REPORT_SCHEMA)
