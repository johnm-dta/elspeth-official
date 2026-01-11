"""Tests for CheckpointManager."""

from elspeth.core.sda.checkpoint import CheckpointManager


def test_checkpoint_manager_loads_existing_ids(tmp_path):
    """CheckpointManager loads existing checkpoint IDs."""
    checkpoint_path = tmp_path / "checkpoint.txt"

    # Create checkpoint with 2 IDs (plain text format)
    with checkpoint_path.open("w", encoding="utf-8") as f:
        f.write("row1\n")
        f.write("row2\n")

    manager = CheckpointManager(checkpoint_path, field="id")

    assert manager.is_processed("row1")
    assert manager.is_processed("row2")
    assert not manager.is_processed("row3")


def test_checkpoint_manager_marks_processed(tmp_path):
    """CheckpointManager appends new IDs to checkpoint."""
    checkpoint_path = tmp_path / "checkpoint.txt"

    manager = CheckpointManager(checkpoint_path, field="id")
    manager.mark_processed("row1")
    manager.mark_processed("row2")

    # Verify file was written
    assert checkpoint_path.exists()

    # Verify IDs are tracked
    assert manager.is_processed("row1")
    assert manager.is_processed("row2")

    # Verify new manager loads same IDs
    manager2 = CheckpointManager(checkpoint_path, field="id")
    assert manager2.is_processed("row1")
    assert manager2.is_processed("row2")
