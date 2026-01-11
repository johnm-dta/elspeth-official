"""Tests for LLM executor landscape integration."""

import json

from elspeth.core.landscape import RunLandscape


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str = "test response"):
        self.response = response
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str, metadata: dict | None = None) -> dict:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return {"content": self.response}


class TestLLMExecutorLandscape:
    """Test LLM executor logs calls to landscape."""

    def test_executor_logs_llm_calls(self):
        """LLMExecutor logs request/response to landscape."""
        from elspeth.core.sda.llm_executor import LLMExecutor

        client = MockLLMClient(response="Hello!")

        with RunLandscape(capture_llm_calls=True) as landscape:
            executor = LLMExecutor(
                llm_client=client,
                middlewares=[],
                retry_config=None,
                rate_limiter=None,
                cost_tracker=None,
            )
            result = executor.execute(
                system_prompt="You are helpful",
                user_prompt="Say hello",
                metadata={"row_id": 1},
            )

            assert result["content"] == "Hello!"

            # Check LLM call was logged
            calls_dir = landscape.root / "intermediate" / "llm_calls"
            call_files = list(calls_dir.glob("*.json"))
            assert len(call_files) == 1

            with call_files[0].open() as f:
                logged = json.load(f)

            assert logged["request"]["system_prompt"] == "You are helpful"
            assert logged["request"]["user_prompt"] == "Say hello"
            assert logged["response"]["content"] == "Hello!"
            assert logged["metadata"]["row_id"] == 1

    def test_executor_works_without_landscape(self):
        """LLMExecutor works when no landscape active."""
        from elspeth.core.sda.llm_executor import LLMExecutor

        client = MockLLMClient(response="Hi")
        executor = LLMExecutor(
            llm_client=client,
            middlewares=[],
            retry_config=None,
            rate_limiter=None,
            cost_tracker=None,
        )
        result = executor.execute(
            system_prompt="Be helpful",
            user_prompt="Hello",
            metadata={},
        )

        assert result["content"] == "Hi"

    def test_executor_skips_logging_when_disabled(self):
        """LLMExecutor respects capture_llm_calls=False."""
        from elspeth.core.sda.llm_executor import LLMExecutor

        client = MockLLMClient()

        with RunLandscape(capture_llm_calls=False) as landscape:
            executor = LLMExecutor(
                llm_client=client,
                middlewares=[],
                retry_config=None,
                rate_limiter=None,
                cost_tracker=None,
            )
            executor.execute(
                system_prompt="System",
                user_prompt="User",
                metadata={},
            )

            calls_dir = landscape.root / "intermediate" / "llm_calls"
            assert not calls_dir.exists() or not list(calls_dir.iterdir())
