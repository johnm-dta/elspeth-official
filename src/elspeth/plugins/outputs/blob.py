"""Azure Blob Storage result sink implementation."""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elspeth.core.interfaces import Artifact, ResultSink
from elspeth.datasources.blob_store import BlobConfig, load_blob_config

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dataclasses when azure libs absent
    from azure.storage.blob import ContentSettings
except Exception:  # pragma: no cover
    ContentSettings = None


class BlobResultSink(ResultSink):
    """Persist experiment payloads to Azure Blob Storage.

    The sink reuses the existing blob configuration files used by datasources so
    operators can target workspace datastores or ad-hoc storage accounts.  The
    uploaded assets include a JSON payload of the experiment results and an
    optional manifest describing auxiliary metadata.
    """

    def __init__(
        self,
        *,
        config_path: str | Path,
        profile: str = "default",
        path_template: str | None = None,
        filename: str = "results.json",
        manifest_template: str | None = None,
        manifest_suffix: str = ".manifest.json",
        include_manifest: bool = True,
        overwrite: bool = True,
        credential: Any | None = None,
        credential_env: str | None = None,
        content_type: str = "application/json",
        metadata: Mapping[str, Any] | None = None,
        upload_chunk_size: int = 4 * 1024 * 1024,
        on_error: str = "abort",
    ) -> None:
        self.config = load_blob_config(config_path, profile)
        self.path_template = path_template
        self.filename = filename
        self.manifest_template = manifest_template
        self.manifest_suffix = manifest_suffix
        self.include_manifest = include_manifest
        self.overwrite = overwrite
        self.credential = credential
        self.credential_env = credential_env
        self.content_type = content_type
        self._blob_metadata = self._normalize_metadata(metadata)
        self.upload_chunk_size = max(int(upload_chunk_size), 0)
        self._blob_service_client = None
        if on_error != "abort":
            raise ValueError("on_error must be 'abort'")
        self.on_error = on_error
        self._artifact_inputs: list[Artifact] = []

    def write(self, results: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        timestamp = datetime.now(UTC)
        context = self._build_context(metadata, timestamp)

        try:
            blob_name = self._resolve_blob_name(context)
            if self._artifact_inputs:
                for idx, artifact in enumerate(self._artifact_inputs, start=1):
                    target_name = blob_name if idx == 1 else self._append_suffix(blob_name, idx)
                    data = self._artifact_bytes(artifact)
                    content_type = None
                    if artifact.metadata:
                        content_type = artifact.metadata.get("content_type")
                    upload_metadata = self._build_upload_metadata(metadata, artifact)
                    self._upload_bytes(
                        target_name,
                        data,
                        content_type=content_type or self.content_type,
                        upload_metadata=upload_metadata,
                    )
            else:
                payload_bytes = self._serialize(results)
                self._upload_bytes(
                    blob_name,
                    payload_bytes,
                    content_type=self.content_type,
                    upload_metadata=self._build_upload_metadata(metadata, None),
                )

                if self.include_manifest:
                    manifest_blob = self._resolve_manifest_name(blob_name, context)
                    manifest_payload = self._build_manifest(results, metadata, blob_name, timestamp)
                    manifest_bytes = json.dumps(manifest_payload, indent=2, sort_keys=True).encode("utf-8")
                    self._upload_bytes(
                        manifest_blob,
                        manifest_bytes,
                        content_type="application/json",
                        upload_metadata=self._build_upload_metadata(metadata, None),
                    )
        except Exception:
            raise
        finally:
            self._artifact_inputs = []

    # ------------------------------------------------------------------ helpers
    def _build_context(self, metadata: Mapping[str, Any], timestamp: datetime) -> dict[str, Any]:
        context = {k: v for k, v in metadata.items() if isinstance(k, str)}
        context.setdefault("timestamp", timestamp.strftime("%Y%m%dT%H%M%SZ"))
        context.setdefault("date", timestamp.strftime("%Y-%m-%d"))
        context.setdefault("time", timestamp.strftime("%H%M%S"))
        context.setdefault("blob_path", self.config.blob_path)
        context.setdefault("container", self.config.container_name)
        context.setdefault("filename", self.filename)
        return context

    def _resolve_blob_name(self, context: Mapping[str, Any]) -> str:
        if self.path_template:
            try:
                path = self.path_template.format(**context)
            except KeyError as exc:  # pragma: no cover - configuration error
                missing = exc.args[0]
                raise ValueError(f"Missing placeholder '{missing}' in blob path template") from exc
        else:
            path = self.config.blob_path or self.filename

        if path.endswith("/"):
            path = f"{path}{self.filename}"
        elif not (Path(path).suffix if path else ""):
            path = f"{path.rstrip('/')}/{self.filename}" if path else self.filename
        if not path:
            raise ValueError("Resolved blob path is empty")
        return path

    def _resolve_manifest_name(self, blob_name: str, context: Mapping[str, Any]) -> str:
        if not self.include_manifest:
            return ""
        if self.manifest_template:
            try:
                manifest = self.manifest_template.format(**context)
            except KeyError as exc:  # pragma: no cover - configuration error
                missing = exc.args[0]
                raise ValueError(f"Missing placeholder '{missing}' in manifest template") from exc
            if manifest.endswith("/"):
                manifest = f"{manifest}{self.filename}{self.manifest_suffix}"
            return manifest
        target = Path(blob_name).stem or blob_name
        return f"{target}{self.manifest_suffix}"

    @staticmethod
    def _serialize(results: dict[str, Any]) -> bytes:
        return json.dumps(results, indent=2, sort_keys=True).encode("utf-8")

    def _build_manifest(
        self,
        results: dict[str, Any],
        metadata: Mapping[str, Any],
        blob_name: str,
        timestamp: datetime,
    ) -> dict[str, Any]:
        manifest = {
            "generated_at": timestamp.isoformat(),
            "blob": blob_name,
            "container": self.config.container_name,
            "rows": len(results.get("results", [])),
            "metadata": dict(metadata),
        }
        if "aggregates" in results:
            manifest["aggregates"] = results["aggregates"]
        if "cost_summary" in results:
            manifest["cost_summary"] = results["cost_summary"]
        return manifest

    def _create_blob_client(self, blob_name: str):  # type: ignore[return-any]
        service = self._get_service_client()
        container_client = service.get_container_client(self.config.container_name)
        return container_client.get_blob_client(blob_name)

    def _upload_bytes(
        self,
        blob_name: str,
        data: bytes,
        *,
        content_type: str,
        upload_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        client = self._create_blob_client(blob_name)
        metadata = self._merge_upload_metadata(upload_metadata)
        chunk_size = self.upload_chunk_size
        if not self.overwrite and hasattr(client, "exists") and client.exists():  # pragma: no cover - network call
            raise FileExistsError(f"Blob '{blob_name}' already exists and overwrite is disabled")
        if chunk_size and len(data) > chunk_size and hasattr(client, "stage_block"):
            block_ids = []
            for index in range(0, len(data), chunk_size):
                chunk = data[index : index + chunk_size]
                block_id = base64.b64encode(f"{index // chunk_size:05d}".encode()).decode("ascii")
                client.stage_block(block_id, chunk)
                block_ids.append(block_id)
            settings = None
            if ContentSettings is not None:
                settings = ContentSettings(content_type=content_type)
            client.commit_block_list(block_ids, metadata=metadata, content_settings=settings)
        else:
            client.upload_blob(
                data,
                overwrite=self.overwrite,
                content_type=content_type,
                metadata=metadata,
            )

    def _merge_upload_metadata(self, upload_metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
        combined = dict(self._blob_metadata)
        if upload_metadata:
            for key, value in upload_metadata.items():
                if value is not None:
                    combined[key] = value
        return combined or None

    def _get_service_client(self):  # type: ignore[return-any]
        if self._blob_service_client is not None:
            return self._blob_service_client

        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:  # pragma: no cover - optional dependency missing
            raise RuntimeError("azure-storage-blob is required for BlobResultSink") from exc

        credential = self._resolve_credential(self.config)
        self._blob_service_client = BlobServiceClient(
            account_url=self.config.account_url,
            credential=credential,
        )
        return self._blob_service_client

    def _resolve_credential(self, config: BlobConfig):
        if self.credential is not None:
            return self.credential
        if self.credential_env:
            value = os.getenv(self.credential_env)
            if value:
                return value
        if config.sas_token:
            return config.sas_token
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError:  # pragma: no cover - optional dependency missing
            return None
        return DefaultAzureCredential()

    # Artifact contract ---------------------------------------------------
    def produces(self):  # pragma: no cover - to be overridden when chaining enabled
        return []

    def consumes(self):  # pragma: no cover - to be overridden when chaining enabled
        return []

    def finalize(self, artifacts, *, metadata=None):  # pragma: no cover - optional cleanup
        return None

    @staticmethod
    def _normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, str]:
        if not metadata:
            return {}
        normalized: dict[str, str] = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError("Blob metadata keys must be strings")
            normalized[key] = "" if value is None else str(value)
        return normalized

    def prepare_artifacts(self, artifacts: Mapping[str, list[Artifact]]):  # pragma: no cover - optional
        collected: list[Artifact] = []
        for values in artifacts.values():
            if values:
                collected.extend(values)
        self._artifact_inputs = collected

    @staticmethod
    def _artifact_bytes(artifact: Artifact) -> bytes:
        if artifact.path:
            return Path(artifact.path).read_bytes()
        if artifact.payload is not None:
            payload = artifact.payload
            if isinstance(payload, (bytes, bytearray)):
                return bytes(payload)
            if hasattr(payload, "read"):
                return payload.read()
            return json.dumps(payload).encode("utf-8")
        raise ValueError("Artifact is missing payload data")

    @staticmethod
    def _append_suffix(blob_name: str, index: int) -> str:
        path = Path(blob_name)
        stem = path.stem
        suffix = path.suffix
        parent = str(path.parent) if str(path.parent) != "." else ""
        name = f"{stem}_{index}{suffix}"
        return f"{parent}/{name}" if parent else name

    def _build_upload_metadata(
        self,
        execution_metadata: Mapping[str, Any] | None,
        artifact: Artifact | None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if execution_metadata and execution_metadata.get("security_level"):
            metadata["security_level"] = execution_metadata["security_level"]
        if artifact and artifact.security_level:
            metadata["security_level"] = artifact.security_level
        return metadata


# --- Plugin Registration ---
from elspeth.core.registry import ARTIFACTS_SECTION_SCHEMA, ON_ERROR_ENUM, registry

AZURE_BLOB_SCHEMA = {
    "type": "object",
    "properties": {
        "config_path": {"type": "string"},
        "profile": {"type": "string"},
        "path_template": {"type": "string"},
        "filename": {"type": "string"},
        "manifest_template": {"type": "string"},
        "manifest_suffix": {"type": "string"},
        "include_manifest": {"type": "boolean"},
        "overwrite": {"type": "boolean"},
        "metadata": {"type": "object"},
        "upload_chunk_size": {"type": "integer", "minimum": 0},
        "credential": {},
        "credential_env": {"type": "string"},
        "content_type": {"type": "string"},
        "artifacts": ARTIFACTS_SECTION_SCHEMA,
        "security_level": {"type": "string"},
        "on_error": ON_ERROR_ENUM,
    },
    "required": ["config_path"],
    "additionalProperties": True,
}

registry.register_sink("azure_blob", BlobResultSink, AZURE_BLOB_SCHEMA)
