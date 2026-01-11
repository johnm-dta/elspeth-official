"""Tests for cost tracker implementations."""

from elspeth.core.controls.cost_tracker import FixedPriceCostTracker


def test_fixed_price_cost_tracker_records_usage():
    tracker = FixedPriceCostTracker(prompt_token_price=0.01, completion_token_price=0.02)

    response = {
        "raw": {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }
        }
    }
    metrics = tracker.record(response)

    assert metrics["prompt_tokens"] == 10
    assert metrics["completion_tokens"] == 5
    assert metrics["cost"] == 10 * 0.01 + 5 * 0.02

    summary = tracker.summary()
    assert summary["prompt_tokens"] == 10
    assert summary["completion_tokens"] == 5
    assert summary["total_cost"] == metrics["cost"]
