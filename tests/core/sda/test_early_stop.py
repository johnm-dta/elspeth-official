"""Tests for EarlyStopCoordinator."""

from elspeth.core.sda.early_stop import EarlyStopCoordinator


class MockHaltCondition:
    """Mock halt condition plugin."""

    def __init__(self, should_halt: bool = False, reason: dict | None = None):
        self.should_halt = should_halt
        self.reason = reason or {"reason": "test halt"}
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True

    def check(self, record: dict, metadata: dict | None = None) -> dict | None:
        if self.should_halt:
            return self.reason
        return None


def test_coordinator_initializes_plugins():
    """EarlyStopCoordinator initializes halt condition plugins."""
    plugin = MockHaltCondition()
    coordinator = EarlyStopCoordinator(plugins=[plugin])

    assert plugin.reset_called is True
    assert not coordinator.is_stopped()


def test_coordinator_detects_halt_condition():
    """EarlyStopCoordinator detects when halt condition is met."""
    plugin = MockHaltCondition(should_halt=True, reason={"reason": "budget exceeded"})
    coordinator = EarlyStopCoordinator(plugins=[plugin])

    # Initially not stopped
    assert not coordinator.is_stopped()

    # Check record that triggers halt
    coordinator.check_record({"cost": 100}, row_index=5)

    # Now stopped
    assert coordinator.is_stopped()
    reason = coordinator.get_reason()
    assert reason["reason"] == "budget exceeded"
    assert reason["row_index"] == 5
