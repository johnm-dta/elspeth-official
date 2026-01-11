"""Result sink that archives experiment outputs into a ZIP bundle."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from elspeth.core.interfaces import Artifact, ArtifactDescriptor, ResultSink
from elspeth.core.security import normalize_security_level, resolve_security_level

logger = logging.getLogger(__name__)


class ZipResultSink(ResultSink):
    """Bundle results, manifest, and optional CSV into a compressed archive."""

    def __init__(
        self,
        *,
        base_path: str | Path,
        bundle_name: str | None = None,
        timestamped: bool = True,
        include_manifest: bool = True,
        include_results: bool = True,
        include_csv: bool = False,
        manifest_name: str = "manifest.json",
        results_name: str = "results.json",
        csv_name: str = "results.csv",
        on_error: str = "abort",
    ) -> None:
        self.base_path = Path(base_path)
        self.bundle_name = bundle_name
        self.timestamped = timestamped
        self.include_manifest = include_manifest
        self.include_results = include_results
        self.include_csv = include_csv
        self.manifest_name = manifest_name
        self.results_name = results_name
        self.csv_name = csv_name
        if on_error != "abort":
            raise ValueError("on_error must be 'abort'")
        self.on_error = on_error
        self._last_archive_path: str | None = None
        self._last_artifacts: dict[str, Any] = {}
        self._additional_inputs: dict[str, list[Artifact]] = {}
        self._security_level: str | None = None

    def write(self, results: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        timestamp = datetime.now(UTC)
        try:
            archive_path = self._resolve_path(metadata, timestamp)
            archive_path.parent.mkdir(parents=True, exist_ok=True)

            with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as bundle:
                if self.include_results:
                    payload = json.dumps(results, indent=2, sort_keys=True)
                    bundle.writestr(self.results_name, payload)

                if self.include_manifest:
                    manifest = self._build_manifest(results, metadata, timestamp)
                    bundle.writestr(self.manifest_name, json.dumps(manifest, indent=2, sort_keys=True))

                if self.include_csv:
                    csv_data = self._render_csv(results)
                    bundle.writestr(self.csv_name, csv_data)

                # Include upstream artifacts
                counter = 0
                for _key, artifacts in self._additional_inputs.items():
                    for artifact in artifacts:
                        counter += 1
                        name = None
                        if artifact.metadata:
                            name = artifact.metadata.get("filename") or artifact.metadata.get("path")
                            if name and Path(name).is_absolute():
                                name = Path(name).name
                        if not name and artifact.path:
                            name = Path(artifact.path).name
                        if not name:
                            name = f"artifact_{counter}"
                        data = self._read_artifact(artifact)
                        bundle.writestr(name, data)
            self._last_archive_path = str(archive_path)
            self._last_artifacts = {
                "results": self.results_name if self.include_results else None,
                "manifest": self.manifest_name if self.include_manifest else None,
                "csv": self.csv_name if self.include_csv else None,
            }
            if metadata:
                self._security_level = normalize_security_level(metadata.get("security_level"))
        except Exception:
            raise
        finally:
            self._additional_inputs = {}

    # ------------------------------------------------------------------ helpers
    def _resolve_path(self, metadata: Mapping[str, Any], timestamp: datetime) -> Path:
        name = self.bundle_name or str(metadata.get("experiment") or metadata.get("name") or "experiment")
        if self.timestamped:
            name = f"{name}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
        return self.base_path / f"{name}.zip"

    @staticmethod
    def _build_manifest(
        results: Mapping[str, Any],
        metadata: Mapping[str, Any],
        timestamp: datetime,
    ) -> dict[str, Any]:
        manifest = {
            "generated_at": timestamp.isoformat(),
            "rows": len(results.get("results", [])) if isinstance(results.get("results"), list) else 0,
            "metadata": dict(metadata),
        }
        if "aggregates" in results:
            manifest["aggregates"] = results["aggregates"]
        if "cost_summary" in results:
            manifest["cost_summary"] = results["cost_summary"]
        if "failures" in results:
            manifest["failures"] = results["failures"]
        return manifest

    @staticmethod
    def _render_csv(results: Mapping[str, Any]) -> str:
        entries = results.get("results", [])
        if not entries:
            return ""
        rows = []
        for item in entries:
            row_data = item.get("row", {}) if isinstance(item, Mapping) else {}
            response = item.get("response", {}) if isinstance(item, Mapping) else {}
            record = dict(row_data)
            if isinstance(response, Mapping):
                record["llm_content"] = response.get("content")
            responses = item.get("responses") if isinstance(item, Mapping) else None
            if isinstance(responses, Mapping):
                for name, resp in responses.items():
                    if isinstance(resp, Mapping):
                        record[f"llm_{name}"] = resp.get("content")
            rows.append(record)
        df = pd.DataFrame(rows)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue()

    def produces(self):  # pragma: no cover - placeholder for artifact chaining
        return [
            ArtifactDescriptor(name="zip", type="file/zip", persist=True, alias="zip"),
        ]

    def consumes(self):  # pragma: no cover - placeholder for artifact chaining
        return []

    def finalize(self, artifacts, *, metadata=None):  # pragma: no cover - optional cleanup
        return None

    def collect_artifacts(self) -> dict[str, Artifact]:  # pragma: no cover
        if not self._last_archive_path:
            return {}
        metadata = {key: value for key, value in self._last_artifacts.items() if value}
        metadata["security_level"] = self._security_level
        artifact = Artifact(
            id="",
            type="file/zip",
            path=self._last_archive_path,
            metadata=metadata,
            persist=True,
            security_level=self._security_level,
        )
        self._last_archive_path = None
        self._last_artifacts = {}
        self._security_level = None
        return {"zip": artifact}

    def prepare_artifacts(self, artifacts: Mapping[str, list[Artifact]]):  # pragma: no cover
        self._additional_inputs = {key: list(values) for key, values in artifacts.items() if values}
        if not self._security_level and self._additional_inputs:
            levels = [artifact.security_level for values in self._additional_inputs.values() for artifact in values]
            self._security_level = resolve_security_level(*levels)

    @staticmethod
    def _read_artifact(artifact: Artifact) -> bytes:
        if artifact.path:
            return Path(artifact.path).read_bytes()
        if artifact.payload is not None:
            payload = artifact.payload
            if isinstance(payload, (bytes, bytearray)):
                return bytes(payload)
            if hasattr(payload, "read"):
                data = payload.read()
                if isinstance(data, (bytes, bytearray)):
                    return bytes(data)
                if isinstance(data, str):
                    return data.encode("utf-8")
                raise TypeError("Readable payload must return bytes or str")
            return json.dumps(payload).encode("utf-8")
        raise ValueError("Artifact is missing payload data")


# --- Plugin Registration ---
from elspeth.core.registry import ARTIFACTS_SECTION_SCHEMA, ON_ERROR_ENUM, registry

ZIP_BUNDLE_SCHEMA = {
    "type": "object",
    "properties": {
        "base_path": {"type": "string"},
        "bundle_name": {"type": "string"},
        "timestamped": {"type": "boolean"},
        "include_manifest": {"type": "boolean"},
        "include_results": {"type": "boolean"},
        "include_csv": {"type": "boolean"},
        "manifest_name": {"type": "string"},
        "results_name": {"type": "string"},
        "csv_name": {"type": "string"},
        "artifacts": ARTIFACTS_SECTION_SCHEMA,
        "security_level": {"type": "string"},
        "on_error": ON_ERROR_ENUM,
    },
    "required": ["base_path"],
    "additionalProperties": True,
}

registry.register_sink("zip_bundle", ZipResultSink, ZIP_BUNDLE_SCHEMA)

__all__ = ["ZipResultSink"]
