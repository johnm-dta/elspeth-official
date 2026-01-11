"""Tests for orchestrator landscape integration."""

from pathlib import Path
from typing import Any

import pandas as pd

from elspeth.core.landscape import get_current_landscape
from elspeth.core.sda.config import SDACycleConfig, SDASuite
from elspeth.orchestrators.standard import StandardOrchestrator


class CollectingSink:
    """Mock sink that collects payloads."""

    def __init__(self):
        self.payloads: list[dict[str, Any]] = []

    def write(self, payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        self.payloads.append(payload)


def _suite(*cycles: SDACycleConfig) -> SDASuite:
    return SDASuite(root=Path(), cycles=list(cycles))


class TestOrchestratorLandscape:
    """Test orchestrators create and manage landscape."""

    def test_standard_orchestrator_creates_landscape(self):
        """StandardOrchestrator creates landscape for suite run."""
        df = pd.DataFrame([{"text": "hello"}])
        sink = CollectingSink()

        suite = _suite(
            SDACycleConfig(
                name="test",
                temperature=0.1,
                max_tokens=10,
                prompt_system="You are helpful",
                prompt_template="Process: {text}",
            )
        )
        orchestrator = StandardOrchestrator(suite=suite, sinks=[sink])

        defaults = {
            "landscape": {"persist": False, "capture_llm_calls": True},
        }

        # Track landscape during run
        landscape_during_run = []
        original_build_runner = orchestrator.build_runner

        def capturing_build_runner(*args, **kwargs):
            landscape_during_run.append(get_current_landscape())
            return original_build_runner(*args, **kwargs)

        orchestrator.build_runner = capturing_build_runner

        orchestrator.run(df, defaults=defaults)

        # Landscape should have been active during build_runner
        assert len(landscape_during_run) == 1
        assert landscape_during_run[0] is not None

    def test_standard_orchestrator_cleans_up_landscape(self):
        """StandardOrchestrator cleans up landscape after run."""
        df = pd.DataFrame([{"text": "hello"}])
        sink = CollectingSink()

        suite = _suite(
            SDACycleConfig(
                name="test",
                temperature=0.1,
                max_tokens=10,
                prompt_system="You are helpful",
                prompt_template="Process: {text}",
            )
        )
        orchestrator = StandardOrchestrator(suite=suite, sinks=[sink])

        defaults = {
            "landscape": {"persist": False},
        }

        orchestrator.run(df, defaults=defaults)

        # Landscape should be None after run completes
        assert get_current_landscape() is None

    def test_landscape_saves_config(self):
        """Landscape saves resolved config at start."""
        df = pd.DataFrame([{"text": "hello"}])
        sink = CollectingSink()

        suite = _suite(
            SDACycleConfig(
                name="test",
                temperature=0.1,
                max_tokens=10,
                prompt_system="You are helpful",
                prompt_template="Process: {text}",
            )
        )
        orchestrator = StandardOrchestrator(suite=suite, sinks=[sink])

        defaults = {
            "landscape": {"persist": True},  # Persist so we can check
        }

        # Capture landscape root during run
        landscape_root = []
        original_build_runner = orchestrator.build_runner

        def capturing_build_runner(*args, **kwargs):
            ls = get_current_landscape()
            if ls:
                landscape_root.append(ls.root)
            return original_build_runner(*args, **kwargs)

        orchestrator.build_runner = capturing_build_runner

        orchestrator.run(df, defaults=defaults)

        # Check that config was saved
        import shutil
        try:
            assert len(landscape_root) == 1
            root = landscape_root[0]
            assert (root / "config" / "resolved" / "settings.yaml").exists()
        finally:
            if landscape_root:
                shutil.rmtree(landscape_root[0], ignore_errors=True)
