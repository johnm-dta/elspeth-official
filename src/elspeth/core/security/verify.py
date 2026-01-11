"""Verification utilities for signed archive bundles."""

from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from elspeth.core.security.signing import verify_signature


def _verify_rsa_signature(data: bytes, signature_b64: str, public_key_pem: str) -> bool:
    """Verify RSA-SHA256 signature using a PEM public key."""
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa, utils
    except ImportError as err:
        raise ImportError(
            "cryptography package required for RSA verification. "
            "Install with: pip install cryptography"
        ) from err

    try:
        public_key = serialization.load_pem_public_key(
            public_key_pem.encode("utf-8"),
            backend=default_backend(),
        )
        if not isinstance(public_key, rsa.RSAPublicKey):
            return False
        signature = base64.b64decode(signature_b64)
        digest = hashlib.sha256(data).digest()

        # Azure Key Vault RS256 signs the raw digest, so use Prehashed
        public_key.verify(
            signature,
            digest,
            padding.PKCS1v15(),
            utils.Prehashed(hashes.SHA256()),
        )
        return True
    except Exception:
        return False


def verify_bundle(
    bundle_path: str | Path,
    key: str | bytes | None = None,
) -> dict[str, Any]:
    """Verify a signed archive bundle.

    Supports both HMAC (requires key) and RSA (uses embedded public key) signatures.

    Args:
        bundle_path: Path to the .zip archive file
        key: The signing key (required for HMAC, optional for RSA)

    Returns:
        dict with verification results:
            - valid: bool - True if all checks pass
            - archive_valid: bool - Archive signature valid
            - manifest_valid: bool - Manifest signature valid
            - files_valid: bool - All file hashes match
            - errors: list[str] - Any error messages
            - files_checked: int - Number of files verified
            - algorithm: str - Signing algorithm used
    """
    bundle_path = Path(bundle_path)
    base_name = bundle_path.stem
    bundle_dir = bundle_path.parent

    manifest_path = bundle_dir / f"{base_name}.manifest.json"
    signature_path = bundle_dir / f"{base_name}.signature.json"

    result: dict[str, Any] = {
        "valid": False,
        "archive_valid": False,
        "manifest_valid": False,
        "files_valid": False,
        "errors": [],
        "files_checked": 0,
        "algorithm": "unknown",
    }

    # Check files exist
    if not bundle_path.exists():
        result["errors"].append(f"Archive not found: {bundle_path}")
        return result
    if not manifest_path.exists():
        result["errors"].append(f"Manifest not found: {manifest_path}")
        return result
    if not signature_path.exists():
        result["errors"].append(f"Signature not found: {signature_path}")
        return result

    # Load signature file
    try:
        signature_data = json.loads(signature_path.read_text(encoding="utf-8"))
    except Exception as e:
        result["errors"].append(f"Failed to load signature: {e}")
        return result

    algorithm = signature_data.get("algorithm", "hmac-sha256")
    result["algorithm"] = algorithm
    archive_sig = signature_data.get("archive_signature", "")
    manifest_sig = signature_data.get("manifest_signature", "")

    # Read file contents
    manifest_bytes = manifest_path.read_bytes()
    archive_bytes = bundle_path.read_bytes()

    # Verify signatures based on algorithm
    if algorithm == "rsa-sha256":
        # RSA verification uses embedded public key
        public_key_pem = signature_data.get("public_key", "")
        if not public_key_pem:
            result["errors"].append("RSA signature missing public key")
            return result

        result["manifest_valid"] = _verify_rsa_signature(manifest_bytes, manifest_sig, public_key_pem)
        result["archive_valid"] = _verify_rsa_signature(archive_bytes, archive_sig, public_key_pem)

        if not result["manifest_valid"]:
            result["errors"].append("Manifest RSA signature verification FAILED - file may be tampered")
        if not result["archive_valid"]:
            result["errors"].append("Archive RSA signature verification FAILED - file may be tampered")

    else:
        # HMAC verification requires key
        if not key:
            result["errors"].append("HMAC verification requires a signing key")
            return result

        result["manifest_valid"] = verify_signature(manifest_bytes, manifest_sig, key, algorithm)
        result["archive_valid"] = verify_signature(archive_bytes, archive_sig, key, algorithm)

        if not result["manifest_valid"]:
            result["errors"].append("Manifest signature verification FAILED - file may be tampered")
        if not result["archive_valid"]:
            result["errors"].append("Archive signature verification FAILED - file may be tampered")

    # Verify file hashes in manifest
    try:
        manifest_data = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as e:
        result["errors"].append(f"Failed to parse manifest: {e}")
        return result

    files_in_manifest = manifest_data.get("files", [])
    files_checked = 0
    files_valid = True

    with zipfile.ZipFile(bundle_path, "r") as zf:
        for file_entry in files_in_manifest:
            file_path = file_entry.get("path", "")
            expected_hash = file_entry.get("sha256", "")

            if not file_path or not expected_hash:
                continue

            try:
                with zf.open(file_path) as f:
                    actual_hash = hashlib.sha256(f.read()).hexdigest()
                    if actual_hash != expected_hash:
                        files_valid = False
                        result["errors"].append(
                            f"Hash mismatch for {file_path}: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                        )
                    files_checked += 1
            except KeyError:
                files_valid = False
                result["errors"].append(f"File missing from archive: {file_path}")

    result["files_valid"] = files_valid
    result["files_checked"] = files_checked
    result["valid"] = result["archive_valid"] and result["manifest_valid"] and result["files_valid"]

    return result


def verify_bundle_cli(bundle_path: str, key: str | None = None) -> None:
    """CLI wrapper for bundle verification."""
    result = verify_bundle(bundle_path, key)
    algorithm = result.get("algorithm", "unknown")

    print(f"\n{'='*60}")
    print(f"Bundle Verification: {bundle_path}")
    print(f"{'='*60}")
    print(f"  Algorithm: {algorithm.upper()}")
    print(f"  Archive signature: {'✓ VALID' if result['archive_valid'] else '✗ INVALID'}")
    print(f"  Manifest signature: {'✓ VALID' if result['manifest_valid'] else '✗ INVALID'}")
    print(f"  File hashes ({result['files_checked']} files): {'✓ VALID' if result['files_valid'] else '✗ INVALID'}")
    print(f"{'='*60}")

    if result["errors"]:
        print("\nErrors:")
        for err in result["errors"]:
            print(f"  - {err}")

    print(f"\nOverall: {'✓ BUNDLE VERIFIED' if result['valid'] else '✗ VERIFICATION FAILED'}")
    print()
