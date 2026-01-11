"""Result sink that writes results to a local CSV file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from elspeth.core.interfaces import Artifact, ArtifactDescriptor, ResultSink
from elspeth.core.security import normalize_security_level

logger = logging.getLogger(__name__)


class CsvResultSink(ResultSink):
    def __init__(self, *, path: str, overwrite: bool = True, on_error: str = "abort"):
        self.path = Path(path)
        self.overwrite = overwrite
        if on_error != "abort":
            raise ValueError("on_error must be 'abort'")
        self.on_error = on_error
        self._last_written_path: str | None = None
        self._security_level: str | None = None

    def write(self, results: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        try:
            entries = results.get("results", [])
            if not entries:
                df = pd.DataFrame()
            else:
                rows = []
                for item in entries:
                    row = item.get("row", {})
                    response = item.get("response", {})
                    record = {**row, "llm_content": response.get("content")}
                    responses = item.get("responses") or {}
                    for name, resp in responses.items():
                        record[f"llm_{name}"] = resp.get("content")
                    rows.append(record)
                df = pd.DataFrame(rows)
            if self.path.exists() and not self.overwrite:
                raise FileExistsError(f"CSV sink destination exists: {self.path}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.path, index=False)
            self._last_written_path = str(self.path)
            if metadata:
                self._security_level = normalize_security_level(metadata.get("security_level"))
        except Exception:
            raise

    def produces(self):  # pragma: no cover - placeholder for artifact chaining
        return [
            ArtifactDescriptor(name="csv", type="file/csv", persist=True, alias="csv"),
        ]

    def consumes(self):  # pragma: no cover - placeholder for artifact chaining
        return []

    def finalize(self, artifacts, *, metadata=None):  # pragma: no cover - optional cleanup
        return None

    def collect_artifacts(self) -> dict[str, Artifact]:  # pragma: no cover - optional
        if not self._last_written_path:
            return {}
        artifact = Artifact(
            id="",
            type="file/csv",
            path=self._last_written_path,
            metadata={
                "path": self._last_written_path,
                "content_type": "text/csv",
                "security_level": self._security_level,
            },
            persist=True,
            security_level=self._security_level,
        )
        self._last_written_path = None
        self._security_level = None
        return {"csv": artifact}


# --- Plugin Registration ---
from elspeth.core.registry import ARTIFACTS_SECTION_SCHEMA, ON_ERROR_ENUM, registry

CSV_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "overwrite": {"type": "boolean"},
        "artifacts": ARTIFACTS_SECTION_SCHEMA,
        "security_level": {"type": "string"},
        "on_error": ON_ERROR_ENUM,
    },
    "required": ["path"],
    "additionalProperties": True,
}

registry.register_sink("csv", CsvResultSink, CSV_SCHEMA)
