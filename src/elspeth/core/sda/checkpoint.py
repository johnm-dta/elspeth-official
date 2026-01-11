"""Checkpoint management for resumable SDA execution."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint state for resumable processing.

    Stores full result records as JSONL, enabling recovery of processed
    results if a batch run is interrupted.

    Supports config fingerprinting: if config/prompts change, the checkpoint
    is automatically invalidated to prevent stale results.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        field: str,
        save_results: bool = True,
        config_fingerprint: str | None = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint file (JSONL format)
            field: Field name containing unique row ID
            save_results: If True, save full result records (default).
                         If False, only save row IDs (legacy mode).
            config_fingerprint: SHA256 hash of config files. If provided and
                               doesn't match stored fingerprint, checkpoint
                               is cleared.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.meta_path = self.checkpoint_path.with_suffix(
            self.checkpoint_path.suffix + ".meta"
        )
        self.field = field
        self.save_results = save_results
        self.config_fingerprint = config_fingerprint
        self._processed_ids: set[str] = set()
        self._saved_results: dict[str, dict[str, Any]] = {}
        self._meta_written = False

        # Validate fingerprint before loading
        if config_fingerprint and not self._validate_fingerprint():
            self._clear_checkpoint()
        else:
            self._load_checkpoint()

    def _validate_fingerprint(self) -> bool:
        """Check if stored fingerprint matches current config.

        Returns:
            True if fingerprints match or no stored fingerprint.
            False if mismatch (checkpoint should be cleared).
        """
        if not self.meta_path.exists():
            if self.checkpoint_path.exists():
                # Legacy checkpoint without meta - can't verify, must clear
                logger.warning(
                    "Checkpoint exists without meta file - clearing stale checkpoint"
                )
                return False
            return True  # No checkpoint exists yet

        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            stored_fp = meta.get("fingerprint")

            if stored_fp != self.config_fingerprint:
                logger.warning(
                    "Config changed since last run (fingerprint mismatch) - "
                    "clearing checkpoint to re-process with new config"
                )
                return False
            return True

        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read checkpoint meta: %s - clearing", e)
            return False

    def _clear_checkpoint(self) -> None:
        """Delete checkpoint and meta files."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Cleared stale checkpoint: %s", self.checkpoint_path)
        if self.meta_path.exists():
            self.meta_path.unlink()

    def _save_meta(self) -> None:
        """Write meta file with current fingerprint."""
        if self._meta_written or not self.config_fingerprint:
            return

        meta = {
            "fingerprint": self.config_fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._meta_written = True

    def _load_checkpoint(self) -> None:
        """Load checkpoint from file."""
        if not self.checkpoint_path.exists():
            return

        with self.checkpoint_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Try JSONL format first
                if line.startswith("{"):
                    try:
                        record = json.loads(line)
                        row_id = record.get("_checkpoint_id")
                        if row_id:
                            self._processed_ids.add(row_id)
                            # Store result without checkpoint metadata
                            result = {k: v for k, v in record.items() if not k.startswith("_checkpoint")}
                            self._saved_results[row_id] = result
                        continue
                    except json.JSONDecodeError:
                        pass

                # Fall back to plain text (legacy format)
                self._processed_ids.add(line)

    def is_processed(self, row_id: str) -> bool:
        """Check if row ID has been processed."""
        return row_id in self._processed_ids

    def mark_processed(self, row_id: str, result: dict[str, Any] | None = None) -> None:
        """Mark row ID as processed and save result.

        Args:
            row_id: Unique identifier for the row
            result: Full result record to save (optional if save_results=False)
        """
        if row_id in self._processed_ids:
            return

        self._processed_ids.add(row_id)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure meta file exists on first write
        self._save_meta()

        with self.checkpoint_path.open("a", encoding="utf-8") as f:
            if self.save_results and result is not None:
                # Save full result as JSONL
                record = {**result, "_checkpoint_id": row_id}
                self._saved_results[row_id] = result
                f.write(json.dumps(record) + "\n")
            else:
                # Legacy mode: just save row ID
                f.write(f"{row_id}\n")

    def get_saved_results(self) -> list[dict[str, Any]]:
        """Get all saved results from checkpoint.

        Returns:
            List of result records in order they were saved.
        """
        return list(self._saved_results.values())

    def get_saved_result(self, row_id: str) -> dict[str, Any] | None:
        """Get a specific saved result by row ID."""
        return self._saved_results.get(row_id)

    @property
    def processed_count(self) -> int:
        """Number of rows processed."""
        return len(self._processed_ids)

    @property
    def has_saved_results(self) -> bool:
        """Whether checkpoint has saved result data (not just IDs)."""
        return bool(self._saved_results)
