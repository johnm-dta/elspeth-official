"""Azure Key Vault-backed archive bundle with asymmetric signing.

The signing key is stored in Azure Key Vault and never leaves the vault.
Supports automatic key generation if the key doesn't exist.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from elspeth.core.interfaces import Artifact, ArtifactDescriptor, ResultSink
from elspeth.core.landscape import get_current_landscape

logger = logging.getLogger(__name__)


def _get_keyvault_client(vault_url: str):
    """Get Azure Key Vault CryptographyClient."""
    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.keys import KeyClient
    except ImportError as e:
        raise ImportError(
            "Azure Key Vault SDK required. Install with: pip install azure-keyvault-keys azure-identity"
        ) from e

    credential = DefaultAzureCredential()
    return KeyClient(vault_url=vault_url, credential=credential)


def _get_crypto_client(vault_url: str, key_name: str):
    """Get CryptographyClient for a specific key."""
    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.keys.crypto import CryptographyClient
    except ImportError as e:
        raise ImportError(
            "Azure Key Vault SDK required. Install with: pip install azure-keyvault-keys azure-identity"
        ) from e

    credential = DefaultAzureCredential()
    key_id = f"{vault_url}/keys/{key_name}"
    return CryptographyClient(key_id, credential=credential)


@dataclass
class AzureArchiveBundleSink(ResultSink):
    """Archive bundle sink with Azure Key Vault signing.

    Uses RSA-SHA256 asymmetric signing - the private key never leaves Key Vault.
    Verification can be done with the public key (retrievable from Key Vault).
    """

    base_path: Path
    vault_url: str  # e.g., "https://my-vault.vault.azure.net"
    key_name: str = "dmp-archive-signing-key"
    key_size: int = 2048  # RSA key size (2048, 3072, or 4096)
    archive_name: str = "archive_bundle"
    timestamped: bool = True
    project_root: Path = field(default_factory=Path.cwd)
    include_patterns: list[str] = field(default_factory=lambda: ["**/*.yaml", "**/*.json"])
    source_patterns: list[str] = field(default_factory=list)  # Files to include under landscape/
    extra_paths: str | list[str] | None = None
    metadata_dataset_key: str | None = None
    max_part_size_mb: int = 20  # Split archive if larger than this (0 = no splitting). Azure DevOps limit is ~25MB.
    on_error: str = "abort"

    _last_archive: Path | None = field(default=None, init=False, repr=False)
    _archive_parts: list[Path] = field(default_factory=list, init=False, repr=False)
    _key_client: Any = field(default=None, init=False, repr=False)
    _crypto_client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_path = Path(self.base_path)
        self.project_root = Path(self.project_root).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize Key Vault clients
        self._key_client = _get_keyvault_client(self.vault_url)
        self._ensure_key_exists()
        self._crypto_client = _get_crypto_client(self.vault_url, self.key_name)

    def _ensure_key_exists(self) -> None:
        """Create the signing key if it doesn't exist."""
        try:
            self._key_client.get_key(self.key_name)
            logger.info(f"Using existing Key Vault key: {self.key_name}")
        except Exception:
            # Key doesn't exist, create it
            logger.info(f"Creating new RSA-{self.key_size} key in Key Vault: {self.key_name}")
            self._key_client.create_rsa_key(
                name=self.key_name,
                size=self.key_size,
                key_operations=["sign", "verify"],
            )
            logger.info(f"Created Key Vault key: {self.key_name}")

    def _needs_splitting(self, archive_path: Path) -> bool:
        """Check if archive exceeds the max part size."""
        if self.max_part_size_mb <= 0:
            return False
        max_bytes = self.max_part_size_mb * 1024 * 1024
        return archive_path.stat().st_size > max_bytes

    def _split_archive(self, archive_path: Path) -> list[Path]:
        """Split archive using zip -s into multiple parts.

        Returns list of all part files (.z01, .z02, ..., .zip).
        The original archive is replaced by the split parts.
        """
        if not shutil.which("zip"):
            raise RuntimeError(
                "zip command not found. Install with: apt-get install zip"
            )

        # Validate archive exists before attempting split
        if not archive_path.exists():
            raise RuntimeError(
                f"Archive file not found: {archive_path}. "
                "This may indicate a previous step failed to create the archive."
            )

        size_spec = f"{self.max_part_size_mb}m"
        stem = archive_path.stem
        parent = archive_path.parent.resolve()  # Use absolute path for cwd
        archive_name = archive_path.name  # Just the filename

        # zip -s creates split archive in place
        # It modifies the original .zip and creates .z01, .z02, etc.
        # Use just the filename since cwd is set to the parent directory
        result = subprocess.run(
            ["zip", "-s", size_spec, archive_name, "--out", f"{stem}_split.zip"],
            capture_output=True,
            text=True,
            cwd=str(parent),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to split archive (exit code {result.returncode}): "
                f"stdout={result.stdout!r}, stderr={result.stderr!r}"
            )

        # Collect all parts - .z01, .z02, ..., and the final .zip
        parts: list[Path] = []

        # Find numbered parts (.z01, .z02, etc.)
        part_num = 1
        while True:
            part_path = parent / f"{stem}_split.z{part_num:02d}"
            if part_path.exists():
                parts.append(part_path)
                part_num += 1
            else:
                break

        # The main .zip file (contains central directory, must be last)
        split_zip = parent / f"{stem}_split.zip"
        if split_zip.exists():
            parts.append(split_zip)

        # Remove the original unsplit archive
        archive_path.unlink()

        logger.info(f"Split archive into {len(parts)} parts: {[p.name for p in parts]}")
        return parts

    def _sign_data(self, data: bytes) -> str:
        """Sign data using Key Vault RSA-SHA256."""
        try:
            from azure.keyvault.keys.crypto import SignatureAlgorithm
        except ImportError as e:
            raise ImportError("azure-keyvault-keys required") from e

        # Hash the data first (Key Vault signs the hash)
        digest = hashlib.sha256(data).digest()

        # Sign using Key Vault
        result = self._crypto_client.sign(SignatureAlgorithm.rs256, digest)
        return base64.b64encode(result.signature).decode("ascii")

    def _get_public_key_pem(self) -> str:
        """Get the public key in PEM format for verification."""
        key = self._key_client.get_key(self.key_name)
        # Convert JWK to PEM format
        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
        except ImportError:
            # If cryptography not available, return the key ID instead
            return key.id

        # Get the raw key components (Azure SDK returns bytes directly)
        n_bytes = key.key.n if isinstance(key.key.n, bytes) else base64.urlsafe_b64decode(key.key.n + "==")
        e_bytes = key.key.e if isinstance(key.key.e, bytes) else base64.urlsafe_b64decode(key.key.e + "==")
        n = int.from_bytes(n_bytes, "big")
        e = int.from_bytes(e_bytes, "big")

        # Construct the public key
        public_numbers = rsa.RSAPublicNumbers(e, n)
        public_key = public_numbers.public_key(default_backend())

        # Export as PEM
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return pem.decode("ascii")

    def write(self, results: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> None:
        metadata = dict(metadata or {})
        timestamp = datetime.now(UTC)
        self._archive_parts = []

        try:
            files = self._gather_files(metadata)
            if not files:
                logger.warning("AzureArchiveBundleSink found no files to archive; skipping bundle creation")
                return

            archive_path = self._archive_path(timestamp)
            self._create_archive(archive_path, files, results, metadata, timestamp)

            # Check if we need to split the archive
            if self._needs_splitting(archive_path):
                self._archive_parts = self._split_archive(archive_path)
                # Use the .zip part (last in list) as the primary archive reference
                primary_archive = self._archive_parts[-1]
                archive_stem = primary_archive.stem.replace("_split", "")
            else:
                self._archive_parts = [archive_path]
                primary_archive = archive_path
                archive_stem = archive_path.stem

            manifest_path = self.base_path / f"{archive_stem}.manifest.json"
            manifest_bytes = self._write_manifest(
                manifest_path, files, metadata, results, timestamp,
                archive_parts=self._archive_parts,
            )
            signature_path = self.base_path / f"{archive_stem}.signature.json"
            self._write_signature(signature_path, self._archive_parts, manifest_bytes, timestamp)
            self._last_archive = primary_archive

            if len(self._archive_parts) > 1:
                logger.info(f"Created signed split archive bundle: {len(self._archive_parts)} parts")
            else:
                logger.info(f"Created signed archive bundle: {archive_path}")
        except Exception as e:
            if self.on_error == "abort":
                raise
            logger.error(f"Failed to create archive bundle: {e}")

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
                        arcname = match.name
                    add_file(match, arcname)

        # Include source code under landscape/ for audit/replication
        for pattern in self.source_patterns:
            for match in self.project_root.glob(pattern):
                if match.is_file():
                    try:
                        rel = match.relative_to(self.project_root).as_posix()
                        arcname = f"landscape/{rel}"
                    except ValueError:
                        arcname = f"landscape/{match.name}"
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
            landscape.write_manifest()
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

        return f"""# Signed Archive Bundle (Azure Key Vault)

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
- **Signing**: RSA-SHA256 via Azure Key Vault (private key never exported)

## Files Included

{file_list}
  - outputs/results.json (evaluation results)
  - outputs/run_metadata.json (run information)
  - README.md (this file)

## How to Extract the Contents

### Standard Archive (single .zip file)

**On Windows:**
1. Right-click on the .zip file
2. Select "Extract All..."
3. Choose a destination folder
4. Click "Extract"

**On macOS:**
1. Double-click on the .zip file
2. The contents will be extracted to the same folder

**On Linux:**
```bash
unzip {archive_name}.zip -d extracted_contents/
```

### Split Archive (multiple parts: .z01, .z02, ..., .zip)

If the archive was split due to size limits, you'll have multiple files:
- `{archive_name}_split.z01` (first part)
- `{archive_name}_split.z02` (second part)
- ... (additional parts as needed)
- `{archive_name}_split.zip` (final part with central directory)

**On Windows (7-Zip recommended):**
1. Install 7-Zip if not already installed
2. Right-click on the `.zip` file (the final part)
3. Select "7-Zip" → "Extract Here"

**On macOS:**
```bash
# Using zip command (built-in)
zip -s 0 {archive_name}_split.zip --out {archive_name}_combined.zip
unzip {archive_name}_combined.zip -d extracted_contents/
```

**On Linux:**
```bash
# Combine and extract in one step
zip -s 0 {archive_name}_split.zip --out {archive_name}_combined.zip
unzip {archive_name}_combined.zip -d extracted_contents/

# Or if using 7z:
7z x {archive_name}_split.zip -oextracted_contents/
```

**Important:** All split parts must be in the same directory for extraction to work.

## How to Verify the Bundle (Integrity Check)

This bundle is signed using RSA-SHA256 with a key stored in Azure Key Vault.
The private key never leaves Azure - verification uses the public key.

### What you need:
**For standard archives:**
1. The .zip file (this archive)
2. The .manifest.json file (file listing with checksums)
3. The .signature.json file (digital signatures with public key)

**For split archives:**
1. All archive parts (.z01, .z02, ..., .zip)
2. The .manifest.json file (lists all parts with checksums)
3. The .signature.json file (contains signature for each part)

### Using the elspeth CLI:
```bash
# Verify (public key is embedded in signature.json)
elspeth --verify {archive_name}.zip
```

### Using Python (standalone verification):
```python
import json
import hashlib
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# Load signature file
with open("{archive_name}.signature.json") as f:
    sig_data = json.load(f)

# Load public key
public_key_pem = sig_data["public_key"]
from cryptography.hazmat.backends import default_backend
public_key = serialization.load_pem_public_key(
    public_key_pem.encode(), backend=default_backend()
)

# Verify archive signature
with open("{archive_name}.zip", "rb") as f:
    archive_hash = hashlib.sha256(f.read()).digest()

signature = base64.b64decode(sig_data["archive_signature"])
public_key.verify(signature, archive_hash, padding.PKCS1v15(), hashes.SHA256())
print("Archive signature: VALID")
```

### What the verification checks:
1. **Archive signature** - RSA-SHA256 signature of the ZIP file
2. **Manifest signature** - RSA-SHA256 signature of the manifest
3. **File hashes** - SHA-256 checksums of all files inside

### Understanding the results:
- ✓ VALID = The signature is valid, no tampering detected
- ✗ INVALID = Verification failed, possible tampering

If verification fails:
1. Ensure all three files (.zip, .manifest.json, .signature.json) are present
2. Confirm the files weren't corrupted during transfer
3. Contact your administrator if problems persist

## Security Notes

- The signing key is an RSA-{self.key_size} key stored in Azure Key Vault
- The private key NEVER leaves Azure Key Vault
- Only the public key is included in the signature file for verification
- Anyone can verify the signature, but only Key Vault can create new signatures

## File Formats

### results.json
Contains the evaluation results in JSON format.

### run_metadata.json
Contains information about the run including timestamp and row count.

### manifest.json (outside the archive)
Lists all files with SHA-256 checksums.

### signature.json (outside the archive)
Contains RSA-SHA256 signatures and the public key for verification.

## Support

If you have questions about this bundle:
1. Contact your DMP system administrator
2. Reference the archive name: {archive_name}

---
*This README was automatically generated by the DMP Archive Bundle system.*
*Signed with Azure Key Vault*
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
        *,
        archive_parts: list[Path],
    ) -> bytes:
        file_entries = []
        for file_path, arcname in files:
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            file_entries.append({"path": arcname, "sha256": digest})

        # Build archive parts info with hashes
        parts_info = []
        for part in archive_parts:
            part_hash = hashlib.sha256(part.read_bytes()).hexdigest()
            parts_info.append({
                "name": part.name,
                "size": part.stat().st_size,
                "sha256": part_hash,
            })

        manifest: dict[str, Any] = {
            "generated_at": timestamp.isoformat(),
            "files": file_entries,
            "row_count": len(results.get("results", [])),
        }

        # For single archive, keep backward-compatible "archive" field
        if len(archive_parts) == 1:
            manifest["archive"] = archive_parts[0].name
        else:
            # For split archives, list all parts in order
            manifest["archive_parts"] = parts_info
            manifest["split_archive"] = True

        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        return manifest_bytes

    def _write_signature(
        self,
        signature_path: Path,
        archive_parts: list[Path],
        manifest_bytes: bytes,
        timestamp: datetime,
    ) -> None:
        # Sign manifest
        manifest_sig = self._sign_data(manifest_bytes)

        # Get public key for verification
        public_key_pem = self._get_public_key_pem()

        signature: dict[str, Any] = {
            "algorithm": "rsa-sha256",
            "key_vault": self.vault_url,
            "key_name": self.key_name,
            "manifest_signature": manifest_sig,
            "public_key": public_key_pem,
            "generated_at": timestamp.isoformat(),
        }

        if len(archive_parts) == 1:
            # Single archive - backward compatible format
            archive_bytes = archive_parts[0].read_bytes()
            signature["archive"] = archive_parts[0].name
            signature["archive_signature"] = self._sign_data(archive_bytes)
        else:
            # Split archive - sign each part
            signature["split_archive"] = True
            part_signatures = []
            for part in archive_parts:
                part_bytes = part.read_bytes()
                part_sig = self._sign_data(part_bytes)
                part_signatures.append({
                    "name": part.name,
                    "signature": part_sig,
                })
            signature["archive_parts"] = part_signatures

        signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    def produces(self) -> list[ArtifactDescriptor]:
        # Note: For split archives, multiple archive_part artifacts are produced
        return [
            ArtifactDescriptor(name="archive", type="file/zip", alias="archive"),
            ArtifactDescriptor(name="archive_part", type="file/zip-part", alias="archive_part"),
            ArtifactDescriptor(name="manifest", type="file/json", alias="archive_manifest"),
            ArtifactDescriptor(name="signature", type="file/json", alias="archive_signature"),
        ]

    def collect_artifacts(self) -> dict[str, Artifact]:
        if not self._archive_parts:
            return {}

        artifacts: dict[str, Artifact] = {}

        # Determine the base stem (without _split suffix)
        primary = self._archive_parts[-1]  # .zip is last
        base_stem = primary.stem.replace("_split", "")
        base_path = primary.parent

        if len(self._archive_parts) == 1:
            # Single archive
            artifacts["archive"] = Artifact(
                id=f"archive:{primary.name}",
                type="file/zip",
                path=str(primary),
                metadata={"content_type": "application/zip"},
            )
        else:
            # Split archive - return "archive" artifact with all parts in metadata
            # The path points to the primary .zip, but all_parts contains all files
            all_part_paths = [str(p) for p in self._archive_parts]
            artifacts["archive"] = Artifact(
                id=f"archive:{primary.name}",
                type="file/zip",
                path=str(primary),
                metadata={
                    "content_type": "application/zip",
                    "split_archive": True,
                    "all_parts": all_part_paths,
                },
            )

            # Also include individual part artifacts for fine-grained access
            for i, part in enumerate(self._archive_parts):
                # Determine content type based on extension
                if part.suffix == ".zip":
                    content_type = "application/zip"
                    artifact_type = "file/zip"
                else:
                    # .z01, .z02, etc. are binary parts
                    content_type = "application/octet-stream"
                    artifact_type = "file/zip-part"

                artifacts[f"archive_part_{i}"] = Artifact(
                    id=f"archive_part:{part.name}",
                    type=artifact_type,
                    path=str(part),
                    metadata={"content_type": content_type, "part_index": i},
                )

        # Manifest and signature use base stem
        artifacts["manifest"] = Artifact(
            id=f"manifest:{base_stem}",
            type="file/json",
            path=str(base_path / f"{base_stem}.manifest.json"),
            metadata={"content_type": "application/json"},
        )
        artifacts["signature"] = Artifact(
            id=f"signature:{base_stem}",
            type="file/json",
            path=str(base_path / f"{base_stem}.signature.json"),
            metadata={"content_type": "application/json"},
        )

        return artifacts


# --- Plugin Registration ---
from elspeth.core.registry import ARTIFACTS_SECTION_SCHEMA, ON_ERROR_ENUM, registry

AZURE_ARCHIVE_BUNDLE_SCHEMA = {
    "type": "object",
    "properties": {
        "base_path": {"type": "string"},
        "vault_url": {"type": "string"},
        "key_name": {"type": "string"},
        "key_size": {"type": "integer", "enum": [2048, 3072, 4096]},
        "archive_name": {"type": "string"},
        "timestamped": {"type": "boolean"},
        "project_root": {"type": "string"},
        "include_patterns": {"type": "array", "items": {"type": "string"}},
        "source_patterns": {"type": "array", "items": {"type": "string"}},
        "extra_paths": {"type": "array", "items": {"type": "string"}},
        "metadata_dataset_key": {"type": "string"},
        "max_part_size_mb": {"type": "integer", "minimum": 0, "description": "Split archive if larger than this (MB). 0 = no splitting."},
        "artifacts": ARTIFACTS_SECTION_SCHEMA,
        "security_level": {"type": "string"},
        "on_error": ON_ERROR_ENUM,
    },
    "required": ["base_path", "vault_url"],
    "additionalProperties": True,
}

registry.register_sink("azure_archive_bundle", AzureArchiveBundleSink, AZURE_ARCHIVE_BUNDLE_SCHEMA)
