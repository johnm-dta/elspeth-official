"""Tests for LLMExecutor middleware hooks."""

from elspeth.core.sda.llm_executor import LLMExecutor


class RecordingMiddleware:
    def __init__(self):
        self.before = []
        self.after = []
        self.exhausted = []

    def before_request(self, request):
        self.before.append(request.metadata.get("attempt"))
        return request

    def after_response(self, request, response):
        self.after.append(request.metadata.get("attempt"))
        return response

    def on_retry_exhausted(self, request, metadata, error):
        self.exhausted.append(metadata.get("attempts"))


class SucceedingLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, metadata=None):
        self.calls += 1
        return {"content": "ok"}


class FailingLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, metadata=None):
        self.calls += 1
        raise RuntimeError("boom")


def test_middleware_before_after_hooks_invoked():
    mw = RecordingMiddleware()
    llm = SucceedingLLM()
    executor = LLMExecutor(
        llm_client=llm,
        middlewares=[mw],
        retry_config={"max_attempts": 1},
        rate_limiter=None,
        cost_tracker=None,
    )

    executor.execute("user", {"row_id": "1"})

    assert mw.before == [1]
    assert mw.after == [1]
    assert mw.exhausted == []
    assert llm.calls == 1


def test_on_retry_exhausted_hook_invoked():
    mw = RecordingMiddleware()
    llm = FailingLLM()
    executor = LLMExecutor(
        llm_client=llm,
        middlewares=[mw],
        retry_config={"max_attempts": 2, "initial_delay": 0},
        rate_limiter=None,
        cost_tracker=None,
    )

    try:
        executor.execute("user", {"row_id": "1"})
    except RuntimeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected failure to exhaust retries")

    assert mw.exhausted == [2]
    assert mw.before == [1, 2]
    assert mw.after == []
    assert llm.calls == 2
