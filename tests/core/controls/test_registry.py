"""Tests for controls registry validation and creation."""

import pytest

from elspeth.core.controls import registry
from elspeth.core.controls.cost_tracker import FixedPriceCostTracker
from elspeth.core.controls.rate_limit import FixedWindowRateLimiter
from elspeth.core.validation import ConfigurationError


def test_create_rate_limiter_validates_known_plugins():
    limiter = registry.create_rate_limiter({"plugin": "fixed_window", "options": {"requests": 2, "per_seconds": 1}})
    assert isinstance(limiter, FixedWindowRateLimiter)

    with pytest.raises(ValueError):
        registry.create_rate_limiter({"plugin": "unknown"})


def test_validate_rate_limiter_rejects_invalid_options():
    with pytest.raises(ConfigurationError):
        registry.validate_rate_limiter({"plugin": "fixed_window", "options": {"requests": 0}})


def test_create_cost_tracker_validates_known_plugins():
    tracker = registry.create_cost_tracker({"plugin": "fixed_price", "options": {"prompt_token_price": 0.1}})
    assert isinstance(tracker, FixedPriceCostTracker)

    with pytest.raises(ValueError):
        registry.create_cost_tracker({"plugin": "unknown"})
