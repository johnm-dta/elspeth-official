"""Tests for checkpoint fingerprint validation."""

import json

from elspeth.core.sda.checkpoint import CheckpointManager


class TestCheckpointFingerprint:
    """Tests for fingerprint-based checkpoint invalidation."""

    def test_saves_meta_file_on_first_write(self, tmp_path):
        """Meta file created when first result is checkpointed."""
        checkpoint_path = tmp_path / "checkpoint.jsonl"

        manager = CheckpointManager(
            checkpoint_path,
            field="id",
            config_fingerprint="abc123",
        )
        manager.mark_processed("row1", {"id": "row1", "result": "value"})

        meta_path = tmp_path / "checkpoint.jsonl.meta"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["fingerprint"] == "abc123"
        assert "created_at" in meta

    def test_valid_fingerprint_loads_checkpoint(self, tmp_path):
        """Checkpoint loaded when fingerprint matches."""
        checkpoint_path = tmp_path / "checkpoint.jsonl"
        meta_path = tmp_path / "checkpoint.jsonl.meta"

        # Create checkpoint with fingerprint
        checkpoint_path.write_text('{"_checkpoint_id": "row1", "id": "row1", "data": "saved"}\n')
        meta_path.write_text(json.dumps({"fingerprint": "matching123", "created_at": "2025-01-01T00:00:00Z"}))

        manager = CheckpointManager(
            checkpoint_path,
            field="id",
            config_fingerprint="matching123",
        )

        assert manager.is_processed("row1")
        assert manager.processed_count == 1

    def test_mismatched_fingerprint_clears_checkpoint(self, tmp_path):
        """Checkpoint cleared when fingerprint differs."""
        checkpoint_path = tmp_path / "checkpoint.jsonl"
        meta_path = tmp_path / "checkpoint.jsonl.meta"

        # Create checkpoint with old fingerprint
        checkpoint_path.write_text('{"_checkpoint_id": "row1", "id": "row1", "data": "stale"}\n')
        meta_path.write_text(json.dumps({"fingerprint": "old_fingerprint", "created_at": "2025-01-01T00:00:00Z"}))

        manager = CheckpointManager(
            checkpoint_path,
            field="id",
            config_fingerprint="new_fingerprint",  # Different!
        )

        # Checkpoint should be cleared
        assert not manager.is_processed("row1")
        assert manager.processed_count == 0
        assert not checkpoint_path.exists()

    def test_missing_meta_file_creates_new(self, tmp_path):
        """Legacy checkpoint without meta file is cleared and meta created."""
        checkpoint_path = tmp_path / "checkpoint.jsonl"

        # Legacy checkpoint (no meta file)
        checkpoint_path.write_text('{"_checkpoint_id": "row1", "id": "row1"}\n')

        manager = CheckpointManager(
            checkpoint_path,
            field="id",
            config_fingerprint="new_fingerprint",
        )

        # Legacy checkpoint cleared (can't verify fingerprint)
        assert manager.processed_count == 0

    def test_no_fingerprint_skips_validation(self, tmp_path):
        """No fingerprint param means no validation (backward compatible)."""
        checkpoint_path = tmp_path / "checkpoint.jsonl"

        checkpoint_path.write_text('{"_checkpoint_id": "row1", "id": "row1"}\n')

        # No config_fingerprint provided
        manager = CheckpointManager(
            checkpoint_path,
            field="id",
        )

        # Should load normally
        assert manager.is_processed("row1")
        assert manager.processed_count == 1

    def test_fingerprint_updated_on_clear(self, tmp_path):
        """Meta file updated with new fingerprint after clear."""
        checkpoint_path = tmp_path / "checkpoint.jsonl"
        meta_path = tmp_path / "checkpoint.jsonl.meta"

        # Old checkpoint
        checkpoint_path.write_text('{"_checkpoint_id": "old"}\n')
        meta_path.write_text(json.dumps({"fingerprint": "old_fp"}))

        manager = CheckpointManager(
            checkpoint_path,
            field="id",
            config_fingerprint="new_fp",
        )

        # Write new data
        manager.mark_processed("new_row", {"id": "new_row"})

        # Meta should have new fingerprint
        meta = json.loads(meta_path.read_text())
        assert meta["fingerprint"] == "new_fp"
