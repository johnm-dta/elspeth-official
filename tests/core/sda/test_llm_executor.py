"""Tests for LLMExecutor."""

from elspeth.core.sda.llm_executor import LLMExecutor


class MockLLMClient:
    def __init__(self, fail_count: int = 0):
        self.calls = 0
        self.fail_count = fail_count

    def generate(self, system_prompt: str, user_prompt: str, metadata: dict | None = None) -> dict:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise RuntimeError("LLM error")
        return {"content": "success"}


def test_llm_executor_executes_without_retry():
    """LLMExecutor executes LLM call without retry config."""
    llm_client = MockLLMClient()
    executor = LLMExecutor(
        llm_client=llm_client,
        middlewares=[],
        retry_config=None,
        rate_limiter=None,
        cost_tracker=None,
    )

    result = executor.execute("user prompt", {"row_id": "1"}, system_prompt="system")

    assert result["content"] == "success"
    assert llm_client.calls == 1


def test_llm_executor_retries_on_failure():
    """LLMExecutor retries on LLM failure."""
    llm_client = MockLLMClient(fail_count=2)  # Fail first 2 attempts
    executor = LLMExecutor(
        llm_client=llm_client,
        middlewares=[],
        retry_config={
            "max_attempts": 3,
            "backoff_multiplier": 1.0,
            "initial_delay": 0.01,  # Fast for testing
        },
        rate_limiter=None,
        cost_tracker=None,
    )

    result = executor.execute("prompt", {})

    # Should succeed on 3rd attempt
    assert result["content"] == "success"
    assert llm_client.calls == 3
    assert result["retry"]["attempts"] == 3
