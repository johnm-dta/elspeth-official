"""Sink that copies an input artifact to a destination path."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elspeth.core.interfaces import Artifact, ResultSink
from elspeth.core.security import normalize_security_level

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


class FileCopySink(ResultSink):
    def __init__(self, *, destination: str, overwrite: bool = True, on_error: str = "abort") -> None:
        self.destination = Path(destination)
        self.overwrite = overwrite
        if on_error != "abort":
            raise ValueError("on_error must be 'abort'")
        self.on_error = on_error
        self._source_artifact: Artifact | None = None
        self._written_path: Path | None = None
        self._output_type: str | None = None
        self._security_level: str | None = None

    def prepare_artifacts(self, artifacts: Mapping[str, list[Artifact]]):  # pragma: no cover - optional
        self._source_artifact = None
        self._output_type = None
        for values in artifacts.values():
            if values:
                if len(values) > 1:
                    raise ValueError("FileCopySink supports a single input artifact; received multiple")
                self._source_artifact = values[0]
                self._output_type = self._source_artifact.type
                self._security_level = self._source_artifact.security_level
                return

    def write(self, results: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:  # type: ignore[override]
        if not self._source_artifact or not self._source_artifact.path:
            raise ValueError("FileCopySink requires an input artifact; configure artifacts.consumes")

        src_path = Path(self._source_artifact.path)
        if not src_path.exists():
            raise FileNotFoundError(f"Source artifact path not found: {src_path}")

        if self.destination.exists() and not self.overwrite:
            raise FileExistsError(f"Destination exists: {self.destination}")

        self.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, self.destination)
        self._written_path = self.destination
        if metadata and metadata.get("security_level"):
            self._security_level = normalize_security_level(metadata.get("security_level"))

    def collect_artifacts(self) -> dict[str, Artifact]:  # pragma: no cover - optional
        if not self._written_path:
            return {}
        metadata = {
            "source": self._source_artifact.id if self._source_artifact else None,
            "source_path": self._source_artifact.path if self._source_artifact else None,
            "security_level": self._security_level,
        }
        if self._source_artifact and self._source_artifact.metadata:
            source_ct = self._source_artifact.metadata.get("content_type")
            if source_ct:
                metadata["content_type"] = source_ct
        artifact = Artifact(
            id="",
            type=self._output_type or "file/octet-stream",
            path=str(self._written_path),
            metadata=metadata,
            persist=True,
            security_level=self._security_level,
        )
        self._written_path = None
        self._source_artifact = None
        self._output_type = None
        self._security_level = None
        return {"file": artifact}


# --- Plugin Registration ---
from elspeth.core.registry import ARTIFACTS_SECTION_SCHEMA, ON_ERROR_ENUM, registry

FILE_COPY_SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {"type": "string"},
        "overwrite": {"type": "boolean"},
        "artifacts": ARTIFACTS_SECTION_SCHEMA,
        "security_level": {"type": "string"},
        "on_error": ON_ERROR_ENUM,
    },
    "required": ["destination"],
    "additionalProperties": True,
}

registry.register_sink("file_copy", FileCopySink, FILE_COPY_SCHEMA)
