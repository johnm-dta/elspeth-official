"""Config fingerprint calculation for checkpoint invalidation."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_config_fingerprint(paths: list[Path]) -> str:
    """Compute SHA256 fingerprint of config file contents.

    Args:
        paths: List of file paths to include in fingerprint.
               Missing files are skipped silently.

    Returns:
        64-character hex string (SHA256 hash).
    """
    hasher = hashlib.sha256()

    # Sort paths for deterministic ordering
    sorted_paths = sorted(paths, key=lambda p: str(p))

    for path in sorted_paths:
        if path.exists() and path.is_file():
            try:
                content = path.read_bytes()
                # Include path in hash to distinguish files with same content
                hasher.update(str(path).encode("utf-8"))
                hasher.update(b"\x00")  # Separator
                hasher.update(content)
                hasher.update(b"\x00")  # Separator
            except OSError:
                # Skip unreadable files
                pass

    return hasher.hexdigest()
