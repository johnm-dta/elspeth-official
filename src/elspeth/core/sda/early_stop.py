"""Early stopping coordination for SDA execution."""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.core.sda.plugins import HaltConditionPlugin

logger = logging.getLogger(__name__)


class EarlyStopCoordinator:
    """Coordinates early stopping via halt condition plugins."""

    def __init__(self, plugins: list[HaltConditionPlugin] | None = None):
        """
        Initialize early stop coordinator.

        Args:
            plugins: List of halt condition plugins
        """
        self.plugins = plugins or []
        self._event = threading.Event() if self.plugins else None
        self._lock = threading.Lock() if self.plugins else None
        self._reason: dict[str, Any] | None = None

        # Initialize plugins
        for plugin in self.plugins:
            with suppress(AttributeError):
                plugin.reset()

    def is_stopped(self) -> bool:
        """Check if early stop has been triggered."""
        if not self._event:
            return False
        return self._event.is_set()

    def check_record(self, record: dict[str, Any], row_index: int | None = None) -> None:
        """
        Check if record triggers halt condition.

        Args:
            record: Processed record to check
            row_index: Index of row being processed
        """
        if not self._event or self._event.is_set():
            return

        if not self.plugins or self._reason:
            return

        metadata: dict[str, Any] | None = None
        if row_index is not None:
            metadata = {"row_index": row_index}

        def _evaluate() -> None:
            if (self._event and self._event.is_set()) or self._reason:
                return

            for plugin in self.plugins:
                try:
                    reason = plugin.check(record, metadata=metadata)
                except Exception:
                    logger.exception(
                        "Early-stop plugin '%s' raised an unexpected error; continuing",
                        getattr(plugin, "name", "unknown"),
                    )
                    continue

                if not reason:
                    continue

                reason = dict(reason)
                reason.setdefault("plugin", getattr(plugin, "name", "unknown"))
                if metadata:
                    for key, value in metadata.items():
                        reason.setdefault(key, value)

                self._reason = reason
                if self._event:
                    self._event.set()
                logger.info(
                    "Early stop triggered by plugin '%s' (reason: %s)",
                    reason.get("plugin", getattr(plugin, "name", "unknown")),
                    {k: v for k, v in reason.items() if k != "plugin"},
                )
                break

        if self._lock:
            with self._lock:
                _evaluate()
        else:
            _evaluate()

    def get_reason(self) -> dict[str, Any] | None:
        """Get early stop reason if triggered."""
        return dict(self._reason) if self._reason else None
