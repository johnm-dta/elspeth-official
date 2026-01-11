"""Run landscape for orchestrator-managed artifact collection."""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Context variable for implicit landscape access
_current_landscape: contextvars.ContextVar[RunLandscape | None] = contextvars.ContextVar(
    "landscape", default=None
)


def get_current_landscape() -> RunLandscape | None:
    """Get the active landscape, or None if not in a landscape context."""
    return _current_landscape.get()


def set_current_landscape(landscape: RunLandscape | None) -> contextvars.Token:
    """Set the current landscape. Returns token for reset."""
    return _current_landscape.set(landscape)


def reset_landscape(token: contextvars.Token) -> None:
    """Reset landscape to previous value using token."""
    _current_landscape.reset(token)


CATEGORIES = ("inputs", "outputs", "config", "intermediate")


@dataclass
class RunLandscape:
    """Manages a temporary directory structure for run artifacts.

    The landscape provides a structured temp directory for capturing:
    - inputs: Data from datasources
    - outputs: Results from sinks
    - config: Original and resolved configuration
    - intermediate: Checkpoints, LLM calls, transform scratch data
    """

    base_path: Path | None = None
    persist: bool = False
    capture_llm_calls: bool = True
    clean_before_run: bool = False  # Clear landscape directory before starting
    _root: Path | None = field(default=None, init=False)
    _created_at: datetime = field(default_factory=lambda: datetime.now(UTC), init=False)
    _manifest: dict[str, Any] = field(default_factory=dict, init=False)
    _plugin_registry: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _artifacts: list[dict[str, Any]] = field(default_factory=list, init=False)
    _cleaned_up: bool = field(default=False, init=False)
    _context_token: contextvars.Token | None = field(default=None, init=False)
    _llm_call_counter: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        """Create landscape directory structure."""
        if self.base_path:
            self._root = Path(self.base_path)
            # Clean existing landscape if requested
            if self.clean_before_run and self._root.exists():
                logger.info("Cleaning existing landscape at %s", self._root)
                shutil.rmtree(self._root)
            self._root.mkdir(parents=True, exist_ok=True)
        else:
            self._root = Path(tempfile.mkdtemp(prefix="elspeth_run_"))

        # Create category subdirectories
        for category in CATEGORIES:
            (self._root / category).mkdir(exist_ok=True)

        logger.info("Created run landscape at %s", self._root)

    @property
    def root(self) -> Path:
        """Root directory of the landscape."""
        if self._root is None:
            raise RuntimeError("Landscape not initialized")
        return self._root

    def cleanup(self) -> None:
        """Remove the landscape directory (unless persist=True)."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if self.persist:
            logger.info("Persisting landscape at %s", self._root)
            return

        if self._root and self._root.exists():
            shutil.rmtree(self._root)
            logger.info("Cleaned up landscape at %s", self._root)

    def get_path(self, category: str, plugin_id: str, filename: str) -> Path:
        """Get a path for writing. Creates directories as needed.

        Args:
            category: One of 'inputs', 'outputs', 'config', 'intermediate'
            plugin_id: Unique identifier for the plugin instance
            filename: Name of the file to create

        Returns:
            Path to write the file

        Raises:
            ValueError: If category is invalid
            RuntimeError: If directory creation fails (fail-hard)
        """
        if category not in CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {CATEGORIES}")

        plugin_dir = self.root / category / plugin_id
        try:
            plugin_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Failed to create landscape directory {plugin_dir}: {exc}") from exc

        return plugin_dir / filename

    def register_artifact(
        self,
        category: str,
        plugin_id: str,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a written artifact in the manifest.

        Args:
            category: Category the artifact belongs to
            plugin_id: Plugin that created the artifact
            path: Path to the artifact file
            metadata: Optional metadata about the artifact
        """
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        rel_path = path.relative_to(self.root)
        artifact = {
            "category": category,
            "plugin_id": plugin_id,
            "path": str(rel_path),
            "size": path.stat().st_size,
            "sha256": self._hash_file(path),
            "registered_at": datetime.now(UTC).isoformat(),
        }
        if metadata:
            artifact["metadata"] = metadata

        self._artifacts.append(artifact)
        logger.debug("Registered artifact: %s", rel_path)

    def register_plugin(
        self,
        plugin_id: str,
        plugin_name: str,
        config: dict[str, Any],
    ) -> None:
        """Register a plugin instance in the manifest.

        Args:
            plugin_id: Unique identifier for this plugin instance
            plugin_name: Registry name of the plugin (e.g., 'azure_blob')
            config: Plugin configuration
        """
        self._plugin_registry[plugin_id] = {
            "plugin": plugin_name,
            "config": config,
            "registered_at": datetime.now(UTC).isoformat(),
        }

    def get_manifest(self) -> dict[str, Any]:
        """Return the complete manifest of all registered artifacts."""
        return {
            "created_at": self._created_at.isoformat(),
            "landscape_id": f"run_{self._created_at.strftime('%Y%m%dT%H%M%SZ')}",
            "root_path": str(self.root),
            "plugin_registry": dict(self._plugin_registry),
            "artifacts": list(self._artifacts),
        }

    def save_config(
        self,
        original_paths: list[Path],
        resolved: dict[str, Any],
    ) -> None:
        """Capture original and resolved configuration.

        Args:
            original_paths: Paths to original config files to copy
            resolved: The merged/resolved configuration dict
        """
        import yaml

        # Create subdirectories
        original_dir = self.root / "config" / "original"
        resolved_dir = self.root / "config" / "resolved"
        original_dir.mkdir(parents=True, exist_ok=True)
        resolved_dir.mkdir(parents=True, exist_ok=True)

        # Copy original config files
        for src_path in original_paths:
            src_path = Path(src_path)
            if src_path.exists():
                dest = original_dir / src_path.name
                shutil.copy2(src_path, dest)
                self.register_artifact("config", "original", dest, {
                    "source_path": str(src_path),
                })
                logger.debug("Copied original config: %s", src_path.name)

        # Filter out non-serializable objects (like rate_limiter, cost_tracker instances)
        serializable_config = self._filter_serializable(resolved)

        # Write resolved config
        resolved_path = resolved_dir / "settings.yaml"
        with resolved_path.open("w") as f:
            yaml.safe_dump(serializable_config, f, default_flow_style=False, sort_keys=False)
        self.register_artifact("config", "resolved", resolved_path, {
            "type": "resolved_config",
        })
        logger.debug("Saved resolved config")

    @staticmethod
    def _filter_serializable(data: Any) -> Any:
        """Recursively filter dict to keep only YAML-serializable values."""
        if isinstance(data, dict):
            return {
                k: RunLandscape._filter_serializable(v)
                for k, v in data.items()
                if RunLandscape._is_serializable(v)
            }
        if isinstance(data, list):
            return [
                RunLandscape._filter_serializable(item)
                for item in data
                if RunLandscape._is_serializable(item)
            ]
        return data

    @staticmethod
    def _is_serializable(value: Any) -> bool:
        """Check if a value is YAML-serializable."""
        if value is None:
            return True
        if isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(value, (list, tuple)):
            return all(RunLandscape._is_serializable(item) for item in value)
        if isinstance(value, dict):
            return all(
                isinstance(k, str) and RunLandscape._is_serializable(v)
                for k, v in value.items()
            )
        # Non-serializable (objects, functions, etc.)
        return False

    def __enter__(self) -> RunLandscape:
        """Enter context: set as current landscape."""
        self._context_token = set_current_landscape(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: reset landscape and cleanup."""
        if self._context_token is not None:
            reset_landscape(self._context_token)
        self.cleanup()
        return None  # Don't suppress exceptions

    def log_llm_call(
        self,
        call_id: str | None,
        request: dict[str, Any],
        response: dict[str, Any],
        metadata: dict[str, Any],
    ) -> str | None:
        """Log an LLM request/response pair.

        Args:
            call_id: Unique ID for this call (auto-generated if None)
            request: The LLM request data
            response: The LLM response data
            metadata: Additional metadata (row_id, attempt, etc.)

        Returns:
            The call_id used (useful when auto-generated), or None if logging disabled
        """
        if not self.capture_llm_calls:
            return None

        # Thread-safe counter increment and file write
        with self._lock:
            if call_id is None:
                self._llm_call_counter += 1
                call_id = f"call_{self._llm_call_counter:06d}"

            calls_dir = self.root / "intermediate" / "llm_calls"
            calls_dir.mkdir(parents=True, exist_ok=True)

            call_path = calls_dir / f"{call_id}.json"
            call_data = {
                "call_id": call_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "request": request,
                "response": response,
                "metadata": metadata,
            }

            with call_path.open("w") as f:
                json.dump(call_data, f, indent=2, default=str)

        return call_id

    def write_manifest(self) -> Path:
        """Write the manifest to manifest.json at landscape root.

        Returns:
            Path to the manifest file
        """
        manifest_path = self.root / "manifest.json"
        manifest = self.get_manifest()

        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info("Wrote manifest to %s", manifest_path)
        return manifest_path

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Compute SHA256 hash of a file."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
