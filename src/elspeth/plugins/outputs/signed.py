"""Sink that produces locally signed artifacts."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elspeth.core.interfaces import ResultSink
from elspeth.core.security import generate_signature

if TYPE_CHECKING:
    from elspeth.core.security.signing import Algorithm

logger = logging.getLogger(__name__)


@dataclass
class SignedArtifactSink(ResultSink):
    base_path: Path
    bundle_name: str | None = None
    timestamped: bool = True
    results_name: str = "results.json"
    signature_name: str = "signature.json"
    manifest_name: str = "manifest.json"
    algorithm: Algorithm = "hmac-sha256"
    key: str | None = None
    key_env: str | None = "DMP_SIGNING_KEY"
    on_error: str = "abort"

    def __post_init__(self) -> None:
        self.base_path = Path(self.base_path)
        if self.on_error != "abort":
            raise ValueError("on_error must be 'abort'")

    def write(self, results: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        timestamp = datetime.now(UTC)
        try:
            bundle_dir = self._resolve_bundle_dir(metadata, timestamp)
            bundle_dir.mkdir(parents=True, exist_ok=True)

            results_path = bundle_dir / self.results_name
            results_bytes = json.dumps(results, indent=2, sort_keys=True).encode("utf-8")
            results_path.write_bytes(results_bytes)

            key = self._resolve_key()
            signature_value = generate_signature(results_bytes, key, algorithm=self.algorithm)
            signature_payload = {
                "algorithm": self.algorithm,
                "signature": signature_value,
                "generated_at": timestamp.isoformat(),
                "target": self.results_name,
            }
            signature_path = bundle_dir / self.signature_name
            signature_path.write_text(json.dumps(signature_payload, indent=2, sort_keys=True), encoding="utf-8")

            manifest = self._build_manifest(results, metadata, timestamp, signature_value)
            manifest_path = bundle_dir / self.manifest_name
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            raise

    def _resolve_bundle_dir(self, metadata: dict[str, Any], timestamp: datetime) -> Path:
        name = self.bundle_name or str(metadata.get("experiment") or metadata.get("name") or "signed")
        if self.timestamped:
            stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
            name = f"{name}_{stamp}"
        return self.base_path / name

    def _build_manifest(
        self,
        results: dict[str, Any],
        metadata: dict[str, Any],
        timestamp: datetime,
        signature: str,
    ) -> dict[str, Any]:
        digest = self._hash_results(results)
        manifest = {
            "generated_at": timestamp.isoformat(),
            "rows": len(results.get("results", [])),
            "metadata": metadata,
            "signature": {
                "algorithm": self.algorithm,
                "value": signature,
                "target": self.results_name,
            },
            "digest": digest,
        }
        if "aggregates" in results:
            manifest["aggregates"] = results["aggregates"]
        if "cost_summary" in results:
            manifest["cost_summary"] = results["cost_summary"]
        return manifest

    @staticmethod
    def _hash_results(results: dict[str, Any]) -> str:
        import hashlib

        payload = json.dumps(results, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _resolve_key(self) -> str:
        if self.key:
            return self.key
        if self.key_env:
            env_value = os.getenv(self.key_env)
            if env_value:
                self.key = env_value
                return env_value
        raise ValueError("Signing key not provided; set 'key' or environment variable")

    def produces(self):  # pragma: no cover - placeholder for artifact chaining
        return []

    def consumes(self):  # pragma: no cover - placeholder for artifact chaining
        return []

    def finalize(self, artifacts, *, metadata=None):  # pragma: no cover - optional cleanup
        return None


# --- Plugin Registration ---
from elspeth.core.registry import ARTIFACTS_SECTION_SCHEMA, ON_ERROR_ENUM, registry

SIGNED_ARTIFACT_SCHEMA = {
    "type": "object",
    "properties": {
        "base_path": {"type": "string"},
        "bundle_name": {"type": "string"},
        "key": {"type": "string"},
        "key_env": {"type": "string"},
        "hash_algorithm": {"type": "string"},
        "security_level": {"type": "string"},
        "artifacts": ARTIFACTS_SECTION_SCHEMA,
        "on_error": ON_ERROR_ENUM,
    },
    "required": ["base_path"],
    "additionalProperties": True,
}

registry.register_sink("signed_artifact", SignedArtifactSink, SIGNED_ARTIFACT_SCHEMA)
