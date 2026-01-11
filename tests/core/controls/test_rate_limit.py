"""Tests for rate limiter implementations."""


from elspeth.core.controls.rate_limit import AdaptiveRateLimiter, FixedWindowRateLimiter


def test_fixed_window_utilization_increments():
    limiter = FixedWindowRateLimiter(requests=2, per_seconds=10.0)

    with limiter.acquire():
        pass
    assert 0.0 < limiter.utilization() <= 1.0

    with limiter.acquire():
        pass
    # After two calls in same window, utilization should be at or near 1.0
    assert limiter.utilization() >= 0.9


def test_adaptive_rate_limiter_tracks_tokens():
    limiter = AdaptiveRateLimiter(requests_per_minute=10, tokens_per_minute=20, interval_seconds=60.0)
    # Acquire with estimated tokens to set utilization
    with limiter.acquire({"estimated_tokens": 10}):
        pass
    util_after_estimate = limiter.utilization()
    assert util_after_estimate > 0.0

    # Update usage with actual response metrics to increase utilization
    limiter.update_usage({"metrics": {"prompt_tokens": 8, "completion_tokens": 4}}, metadata={})
    assert limiter.utilization() >= util_after_estimate
