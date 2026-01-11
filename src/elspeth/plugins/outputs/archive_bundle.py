"""Result sink that builds a signed archive of project artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import ZIP_DEFLATED, ZipFile

from elspeth.core.interfaces import Artifact, ArtifactDescriptor, ResultSink
from elspeth.core.landscape import get_current_landscape
from elspeth.core.security import generate_signature

if TYPE_CHECKING:
    from elspeth.core.security.signing import Algorithm

logger = logging.getLogger(__name__)


DEFAULT_PATTERNS = [
    "**/*.py",
    "config/**/*.yaml",
    "config/**/*.yml",
    "config/**/*.json",
    "config/**/*.md",
    "config/**/*.csv",
    "config/**/*.toml",
    "config/**/*.ini",
    "config/**/*.txt",
    "dmp/core/prompts/**/*",
]


@dataclass
class ArchiveBundleSink(ResultSink):
    """Create a compressed archive of key project files and sign it.

    The sink gathers files from ``project_root`` matching ``include_patterns`` as
    well as any dataset paths provided via configuration or metadata, writes
    them to a timestamped ZIP archive, emits a manifest with file metadata, and
    signs the archive bytes using the configured symmetric key (HMAC).
    """

    base_path: Path
    archive_name: str = "dmp_project_bundle"
    timestamped: bool = True
    project_root: Path = Path()
    include_patterns: Sequence[str] = field(default_factory=lambda: list(DEFAULT_PATTERNS))
    extra_paths: Sequence[str] | None = None
    metadata_dataset_key: str | None = "dataset_paths"
    algorithm: Algorithm = "hmac-sha256"
    key: str | None = None
    key_env: str | None = "DMP_ARCHIVE_SIGNING_KEY"
    on_error: str = "abort"
    _last_archive: Path | None = field(default=None, init=False)
    _last_manifest: Path | None = field(default=None, init=False)
    _last_signature: Path | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.base_path = Path(self.base_path)
        self.project_root = Path(self.project_root).resolve()
        if self.on_error != "abort":
            raise ValueError("on_error must be 'abort'")
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ResultSink -----------------------------------------------------------------
    def write(self, results: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        metadata = dict(metadata or {})
        timestamp = datetime.now(UTC)
        try:
            self._last_archive = None
            self._last_manifest = None
            self._last_signature = None
            files = self._gather_files(metadata)
            if not files:
                logger.warning("ArchiveBundleSink found no files to archive; skipping bundle creation")
                return

            archive_path = self._archive_path(timestamp)
            self._create_archive(archive_path, files, results, metadata, timestamp)
            manifest_path = archive_path.with_suffix(".manifest.json")
            manifest_bytes = self._write_manifest(manifest_path, files, metadata, results, timestamp, archive_path.name)
            signature_path = archive_path.with_suffix(".signature.json")
            self._write_signature(signature_path, archive_path, manifest_bytes, timestamp)
            self._last_archive = archive_path
            self._last_manifest = manifest_path
            self._last_signature = signature_path
        except Exception:
            raise

    # Helpers --------------------------------------------------------------------
    def _archive_path(self, timestamp: datetime) -> Path:
        if self.timestamped:
            stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
            filename = f"{self.archive_name}_{stamp}.zip"
        else:
            filename = f"{self.archive_name}.zip"
        return self.base_path / filename

    def _gather_files(self, metadata: Mapping[str, Any]) -> list[tuple[Path, str]]:
        collected: list[tuple[Path, str]] = []
        seen: set[str] = set()

        def add_file(path: Path, arcname: str) -> None:
            path = path.resolve()
            if not path.is_file():
                return
            key = path.as_posix()
            if key in seen:
                return
            seen.add(key)
            collected.append((path, arcname))

        def add_path(path: Path, arc_prefix: str) -> None:
            path = path.resolve()
            if path.is_file():
                add_file(path, f"{arc_prefix}/{path.name}")
                return
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        rel = child.relative_to(path)
                        add_file(child, f"{arc_prefix}/{path.name}/{rel.as_posix()}")

        # Include project files
        for pattern in self.include_patterns:
            for match in self.project_root.glob(pattern):
                if match.is_file():
                    try:
                        arcname = match.relative_to(self.project_root).as_posix()
                    except ValueError:
                        # If match is outside project root (unlikely), fall back to filename
                        arcname = match.name
                    add_file(match, arcname)

        # Include explicitly configured extra paths
        extra_iter: Iterable[str]
        if self.extra_paths:
            extra_iter = [self.extra_paths] if isinstance(self.extra_paths, (str, bytes)) else self.extra_paths
            for path_str in extra_iter:
                path = Path(path_str).expanduser().resolve()
                add_path(path, arc_prefix="inputs/extra")

        # Include dataset paths from metadata
        if self.metadata_dataset_key:
            dataset_entries = metadata.get(self.metadata_dataset_key)
            if isinstance(dataset_entries, Sequence) and not isinstance(dataset_entries, (str, bytes)):
                for entry in dataset_entries:
                    path = Path(entry).expanduser().resolve()
                    add_path(path, arc_prefix="inputs")

        # Include landscape artifacts if active
        landscape = get_current_landscape()
        if landscape:
            # Write manifest before gathering
            landscape.write_manifest()

            # Gather all landscape files
            for child in landscape.root.rglob("*"):
                if child.is_file():
                    try:
                        rel = child.relative_to(landscape.root)
                        arcname = f"landscape/{rel.as_posix()}"
                        add_file(child, arcname)
                    except ValueError:
                        pass

        return collected

    def _generate_readme(self, archive_name: str, files: list[tuple[Path, str]], timestamp: datetime) -> str:
        """Generate a README explaining the bundle contents and verification."""
        file_list = "\n".join(f"  - {arcname}" for _, arcname in sorted(files, key=lambda x: x[1]))

        return f"""# Signed Archive Bundle

## What is this?

This is a signed archive bundle created by the DMP (Digital Marketplace) evaluation
system. It contains the complete record of an evaluation run, including:

- **Configuration files** - The settings used for this evaluation
- **Results data** - The output from processing (JSON format)
- **Run metadata** - Information about when and how the run was executed

## Bundle Information

- **Archive name**: {archive_name}
- **Created**: {timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}
- **Format**: ZIP archive with DEFLATE compression

## Files Included

{file_list}
  - outputs/results.json (evaluation results)
  - outputs/run_metadata.json (run information)
  - README.md (this file)

## How to Extract the Contents

### On Windows:
1. Right-click on the .zip file
2. Select "Extract All..."
3. Choose a destination folder
4. Click "Extract"

### On macOS:
1. Double-click on the .zip file
2. The contents will be extracted to the same folder

### On Linux:
```bash
unzip {archive_name}.zip -d extracted_contents/
```

## How to Verify the Bundle (Integrity Check)

This bundle is digitally signed to ensure it hasn't been tampered with. To verify:

### What you need:
1. The .zip file (this archive)
2. The .manifest.json file (file listing with checksums)
3. The .signature.json file (digital signatures)
4. The signing key (provided by your administrator)

### Using the elspeth CLI:
```bash
# Set the verification key
export DMP_ARCHIVE_SIGNING_KEY="your-key-here"

# Run verification
elspeth --verify {archive_name}.zip
```

### What the verification checks:
1. **Archive signature** - Confirms the .zip file hasn't been modified
2. **Manifest signature** - Confirms the file listing hasn't been modified
3. **File hashes** - Confirms each file inside matches its recorded checksum

### Understanding the results:
- ✓ VALID = The check passed, no tampering detected
- ✗ INVALID = The check failed, possible tampering or wrong key

If verification fails:
1. Check you're using the correct signing key
2. Ensure all three files (.zip, .manifest.json, .signature.json) are present
3. Confirm the files weren't corrupted during transfer
4. Contact your administrator if problems persist

## File Formats

### results.json
Contains the evaluation results in JSON format. Open with any text editor or
JSON viewer. Each row in the original data has a corresponding result entry.

### run_metadata.json
Contains information about the run including:
- Timestamp when the run started
- Number of rows processed
- Any error information

### manifest.json (outside the archive)
Lists all files in the archive with their SHA-256 checksums for integrity
verification.

### signature.json (outside the archive)
Contains the digital signatures (HMAC-SHA256) for both the archive and manifest.

## Support

If you have questions about this bundle or need assistance:
1. Contact your DMP system administrator
2. Reference the archive name and timestamp when reporting issues
3. Keep all three files (.zip, .manifest.json, .signature.json) together

---
*This README was automatically generated by the DMP Archive Bundle system.*
"""

    def _create_archive(
        self,
        archive_path: Path,
        files: list[tuple[Path, str]],
        results: Mapping[str, Any],
        metadata: Mapping[str, Any],
        timestamp: datetime,
    ) -> None:
        readme_content = self._generate_readme(archive_path.stem, files, timestamp)

        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            for file_path, arcname in files:
                archive.write(file_path, arcname=arcname)
            archive.writestr(
                "outputs/results.json",
                json.dumps(results, indent=2, sort_keys=True).encode("utf-8"),
            )
            archive.writestr(
                "outputs/run_metadata.json",
                json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
            )
            archive.writestr("README.md", readme_content.encode("utf-8"))

    def _write_manifest(
        self,
        manifest_path: Path,
        files: list[tuple[Path, str]],
        metadata: Mapping[str, Any],
        results: Mapping[str, Any],
        timestamp: datetime,
        archive_name: str,
    ) -> bytes:
        files_section: list[dict[str, Any]] = []
        manifest = {
            "generated_at": timestamp.isoformat(),
            "archive": archive_name,
            "files": files_section,
            "metadata": dict(metadata),
            "results_summary": {
                "rows": len(results.get("results", [])),
                "aggregates": results.get("aggregates"),
                "cost_summary": results.get("cost_summary"),
            },
        }
        for file_path, arcname in files:
            files_section.append(
                {
                    "path": arcname,
                    "size": file_path.stat().st_size,
                    "sha256": self._hash_file(file_path),
                }
            )
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        return manifest_bytes

    def _write_signature(
        self,
        signature_path: Path,
        archive_path: Path,
        manifest_bytes: bytes,
        timestamp: datetime,
    ) -> None:
        key = self._resolve_key()
        archive_bytes = archive_path.read_bytes()
        signature = generate_signature(archive_bytes, key, algorithm=self.algorithm)
        manifest_signature = generate_signature(manifest_bytes, key, algorithm=self.algorithm)
        payload = {
            "algorithm": self.algorithm,
            "generated_at": timestamp.isoformat(),
            "archive": archive_path.name,
            "archive_signature": signature,
            "manifest_signature": manifest_signature,
        }
        signature_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _resolve_key(self) -> str:
        if self.key:
            return self.key
        if self.key_env:
            env_value = os.getenv(self.key_env)
            if env_value:
                self.key = env_value
                return env_value
        raise ValueError("Signing key not provided; set 'key' or environment variable")

    def produces(self):
        return [
            ArtifactDescriptor(name="archive", type="file/zip", persist=True, alias="archive", security_level="official"),  # type: ignore[name-defined]
            ArtifactDescriptor(name="manifest", type="file/json", persist=True, alias="archive_manifest", security_level="official"),  # type: ignore[name-defined]
            ArtifactDescriptor(name="signature", type="file/json", persist=True, alias="archive_signature", security_level="official"),  # type: ignore[name-defined]
        ]

    def consumes(self):
        return []

    def collect_artifacts(self) -> dict[str, Artifact]:
        artifacts: dict[str, Artifact] = {}
        if self._last_archive and self._last_archive.exists():
            artifacts["archive"] = Artifact(
                id=self._last_archive.name,
                type="file/zip",
                path=str(self._last_archive),
                metadata={"description": "Signed archive bundle"},
                persist=True,
            )
        if self._last_manifest and self._last_manifest.exists():
            artifacts["manifest"] = Artifact(
                id=self._last_manifest.name,
                type="file/json",
                path=str(self._last_manifest),
                metadata={"description": "Archive manifest"},
                persist=True,
            )
        if self._last_signature and self._last_signature.exists():
            signature_payload = json.loads(self._last_signature.read_text(encoding="utf-8"))
            artifacts["signature"] = Artifact(
                id=self._last_signature.name,
                type="file/json",
                path=str(self._last_signature),
                metadata={"algorithm": signature_payload.get("algorithm")},
                persist=True,
            )
        return artifacts


# --- Plugin Registration ---
from elspeth.core.registry import ARTIFACTS_SECTION_SCHEMA, ON_ERROR_ENUM, registry

ARCHIVE_BUNDLE_SCHEMA = {
    "type": "object",
    "properties": {
        "base_path": {"type": "string"},
        "archive_name": {"type": "string"},
        "timestamped": {"type": "boolean"},
        "project_root": {"type": "string"},
        "include_patterns": {"type": "array", "items": {"type": "string"}},
        "extra_paths": {"type": "array", "items": {"type": "string"}},
        "metadata_dataset_key": {"type": "string"},
        "algorithm": {"type": "string"},
        "key": {"type": "string"},
        "key_env": {"type": "string"},
        "artifacts": ARTIFACTS_SECTION_SCHEMA,
        "security_level": {"type": "string"},
        "on_error": ON_ERROR_ENUM,
    },
    "required": ["base_path"],
    "additionalProperties": True,
}

registry.register_sink("archive_bundle", ArchiveBundleSink, ARCHIVE_BUNDLE_SCHEMA)
