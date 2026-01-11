"""Registry for rate limiter and cost tracker plugins."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

from elspeth.core.validation import ConfigurationError, validate_schema

from .cost_tracker import CostTracker, FixedPriceCostTracker, NoopCostTracker
from .rate_limit import AdaptiveRateLimiter, FixedWindowRateLimiter, NoopRateLimiter, RateLimiter

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

# Global cache for shared rate limiter instances
# Key: hash of (plugin_name, options), Value: RateLimiter instance
_rate_limiter_cache: dict[str, RateLimiter] = {}


class _Factory:
    def __init__(self, factory: Callable[[dict[str, Any]], Any], schema: Mapping[str, Any] | None = None):
        self.factory = factory
        self.schema = schema

    def validate(self, options: dict[str, Any], *, context: str) -> None:
        if self.schema is None:
            return
        errors = list(validate_schema(options or {}, self.schema, context=context))
        if errors:
            raise ConfigurationError("\n".join(msg.format() for msg in errors))

    def create(self, options: dict[str, Any], *, context: str) -> Any:
        self.validate(options, context=context)
        return self.factory(options)


_rate_limiters: dict[str, _Factory] = {
    "noop": _Factory(lambda options: NoopRateLimiter()),
    "fixed_window": _Factory(
        lambda options: FixedWindowRateLimiter(
            requests=int(options.get("requests", 1)),
            per_seconds=float(options.get("per_seconds", 1.0)),
        ),
        schema={
            "type": "object",
            "properties": {
                "requests": {"type": "integer", "minimum": 1},
                "per_seconds": {"type": "number", "exclusiveMinimum": 0},
            },
            "additionalProperties": True,
        },
    ),
    "adaptive": _Factory(
        lambda options: AdaptiveRateLimiter(
            requests_per_minute=int(options.get("requests_per_minute", options.get("requests", 60)) or 60),
            tokens_per_minute=(lambda value: int(value) if value is not None else None)(options.get("tokens_per_minute")),
            interval_seconds=float(options.get("interval_seconds", 60.0)),
            min_interval=(lambda value: float(value) if value is not None else None)(options.get("min_interval")),
        ),
        schema={
            "type": "object",
            "properties": {
                "requests_per_minute": {"type": "integer", "minimum": 1},
                "requests": {"type": "integer", "minimum": 1},
                "tokens_per_minute": {"type": "integer", "minimum": 0},
                "interval_seconds": {"type": "number", "exclusiveMinimum": 0},
                "min_interval": {"type": "number", "exclusiveMinimum": 0},
            },
            "additionalProperties": True,
        },
    ),
}

_cost_trackers: dict[str, _Factory] = {
    "noop": _Factory(lambda options: NoopCostTracker()),
    "fixed_price": _Factory(
        lambda options: FixedPriceCostTracker(
            prompt_token_price=float(options.get("prompt_token_price", 0.0)),
            completion_token_price=float(options.get("completion_token_price", 0.0)),
        ),
        schema={
            "type": "object",
            "properties": {
                "prompt_token_price": {"type": "number", "minimum": 0},
                "completion_token_price": {"type": "number", "minimum": 0},
            },
            "additionalProperties": True,
        },
    ),
}


def register_rate_limiter(name: str, factory: Callable[[dict[str, Any]], RateLimiter]) -> None:
    _rate_limiters[name] = _Factory(factory)


def register_cost_tracker(name: str, factory: Callable[[dict[str, Any]], CostTracker]) -> None:
    _cost_trackers[name] = _Factory(factory)


def _cache_key(name: str, options: dict[str, Any]) -> str:
    """Generate a cache key for rate limiter config."""
    # Sort keys for consistent hashing
    config_str = json.dumps({"name": name, "options": options}, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def create_rate_limiter(definition: dict[str, Any] | None, *, shared: bool = True) -> RateLimiter | None:
    """Create or retrieve a rate limiter instance.

    Args:
        definition: Rate limiter config dict
        shared: If True, return cached instance for identical configs (default: True)

    Returns:
        RateLimiter instance or None
    """
    if not definition:
        return None
    name = definition.get("plugin") or definition.get("name")
    options = definition.get("options", {})
    if name not in _rate_limiters:
        raise ValueError(f"Unknown rate limiter plugin '{name}'")

    # Check cache for shared instance
    if shared:
        cache_key = _cache_key(name, options)
        if cache_key in _rate_limiter_cache:
            return _rate_limiter_cache[cache_key]

        # Create new instance and cache it
        instance = cast("RateLimiter", _rate_limiters[name].create(options, context=f"rate_limiter:{name}"))
        _rate_limiter_cache[cache_key] = instance
        return instance

    # Create new instance without caching
    return cast("RateLimiter", _rate_limiters[name].create(options, context=f"rate_limiter:{name}"))


def create_cost_tracker(definition: dict[str, Any] | None) -> CostTracker | None:
    if not definition:
        return None
    name = definition.get("plugin") or definition.get("name")
    options = definition.get("options", {})
    if name not in _cost_trackers:
        raise ValueError(f"Unknown cost tracker plugin '{name}'")
    return cast("CostTracker", _cost_trackers[name].create(options, context=f"cost_tracker:{name}"))


def validate_rate_limiter(definition: dict[str, Any] | None) -> None:
    if not definition:
        return
    name = definition.get("plugin") or definition.get("name")
    options = definition.get("options", {})
    if name not in _rate_limiters:
        raise ConfigurationError(f"Unknown rate limiter plugin '{name}'")
    _rate_limiters[name].validate(options, context=f"rate_limiter:{name}")


def validate_cost_tracker(definition: dict[str, Any] | None) -> None:
    if not definition:
        return
    name = definition.get("plugin") or definition.get("name")
    options = definition.get("options", {})
    if name not in _cost_trackers:
        raise ConfigurationError(f"Unknown cost tracker plugin '{name}'")
    _cost_trackers[name].validate(options, context=f"cost_tracker:{name}")


def clear_rate_limiter_cache() -> None:
    """Clear the rate limiter cache. Call between separate runs."""
    _rate_limiter_cache.clear()


__all__ = [
    "clear_rate_limiter_cache",
    "create_cost_tracker",
    "create_rate_limiter",
    "register_cost_tracker",
    "register_rate_limiter",
    "validate_cost_tracker",
    "validate_rate_limiter",
]
