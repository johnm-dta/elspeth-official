"""Tests for LLM middleware implementations."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from elspeth.core.llm.middleware import LLMRequest
from elspeth.plugins.llms.middleware import (
    AuditMiddleware,
    AzureContentSafetyMiddleware,
    HealthMonitorMiddleware,
    PromptShieldMiddleware,
    TracingMiddleware,
)


@pytest.fixture
def sample_request():
    """Create a sample LLM request."""
    return LLMRequest(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is 2 + 2?",
        metadata={"row_id": 123, "experiment": "test"},
    )


@pytest.fixture
def sample_response():
    """Create a sample LLM response."""
    return {
        "content": "The answer is 4.",
        "metrics": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "cost": 0.001,
        },
    }


class TestLLMRequest:
    """Tests for LLMRequest dataclass."""

    def test_create_request(self):
        """Can create an LLM request."""
        request = LLMRequest(
            system_prompt="System",
            user_prompt="User",
            metadata={"key": "value"},
        )
        assert request.system_prompt == "System"
        assert request.user_prompt == "User"
        assert request.metadata == {"key": "value"}

    def test_clone_with_new_user_prompt(self, sample_request):
        """Clone creates new request with modified user_prompt."""
        cloned = sample_request.clone(user_prompt="New prompt")

        assert cloned.user_prompt == "New prompt"
        assert cloned.system_prompt == sample_request.system_prompt
        assert cloned is not sample_request

    def test_clone_with_new_system_prompt(self, sample_request):
        """Clone creates new request with modified system_prompt."""
        cloned = sample_request.clone(system_prompt="New system")

        assert cloned.system_prompt == "New system"
        assert cloned.user_prompt == sample_request.user_prompt

    def test_clone_with_new_metadata(self, sample_request):
        """Clone creates new request with new metadata."""
        cloned = sample_request.clone(metadata={"new": "data"})

        assert cloned.metadata == {"new": "data"}
        assert sample_request.metadata == {"row_id": 123, "experiment": "test"}

    def test_clone_copies_metadata_by_default(self, sample_request):
        """Clone creates independent copy of metadata."""
        cloned = sample_request.clone()
        cloned.metadata["added"] = True

        assert "added" not in sample_request.metadata


class TestAuditMiddleware:
    """Tests for AuditMiddleware."""

    def test_init_defaults(self):
        """Initializes with default values."""
        middleware = AuditMiddleware()

        assert middleware.include_prompts is False
        assert middleware.channel == "dmp.audit"

    def test_init_custom_options(self):
        """Accepts custom options."""
        middleware = AuditMiddleware(
            include_prompts=True,
            channel="custom.audit",
        )

        assert middleware.include_prompts is True
        assert middleware.channel == "custom.audit"

    def test_before_request_returns_request_unchanged(self, sample_request):
        """Before request returns request unchanged."""
        middleware = AuditMiddleware()
        result = middleware.before_request(sample_request)

        assert result is sample_request

    def test_before_request_logs_metadata(self, sample_request, caplog):
        """Before request logs metadata."""
        middleware = AuditMiddleware()
        with caplog.at_level(logging.INFO):
            middleware.before_request(sample_request)

        assert "row_id" in caplog.text
        assert "123" in caplog.text

    def test_before_request_logs_prompts_when_enabled(self, sample_request, caplog):
        """Before request logs prompts when include_prompts is True."""
        middleware = AuditMiddleware(include_prompts=True)
        with caplog.at_level(logging.INFO):
            middleware.before_request(sample_request)

        assert "You are a helpful assistant" in caplog.text
        assert "What is 2 + 2?" in caplog.text

    def test_after_response_returns_response_unchanged(self, sample_request, sample_response):
        """After response returns response unchanged."""
        middleware = AuditMiddleware()
        result = middleware.after_response(sample_request, sample_response)

        assert result is sample_response

    def test_after_response_logs_metrics(self, sample_request, sample_response, caplog):
        """After response logs metrics."""
        middleware = AuditMiddleware()
        with caplog.at_level(logging.INFO):
            middleware.after_response(sample_request, sample_response)

        assert "prompt_tokens" in caplog.text
        assert "20" in caplog.text


class TestPromptShieldMiddleware:
    """Tests for PromptShieldMiddleware."""

    def test_init_defaults(self):
        """Initializes with default values."""
        middleware = PromptShieldMiddleware()

        assert middleware.denied_terms == []
        assert middleware.mask == "[REDACTED]"
        assert middleware.mode == "abort"
        assert middleware.channel == "dmp.prompt_shield"

    def test_init_custom_options(self):
        """Accepts custom options."""
        middleware = PromptShieldMiddleware(
            denied_terms=["secret", "password"],
            mask="***",
            on_violation="mask",
            channel="custom.shield",
        )

        assert middleware.denied_terms == ["secret", "password"]
        assert middleware.mask == "***"
        assert middleware.mode == "mask"

    def test_before_request_passes_clean_prompt(self, sample_request):
        """Clean prompts pass through unchanged."""
        middleware = PromptShieldMiddleware(denied_terms=["forbidden"])
        result = middleware.before_request(sample_request)

        assert result is sample_request

    def test_before_request_aborts_on_blocked_term(self):
        """Aborts when blocked term is found."""
        middleware = PromptShieldMiddleware(
            denied_terms=["secret"],
            on_violation="abort",
        )
        request = LLMRequest(
            system_prompt="System",
            user_prompt="Tell me the secret password",
            metadata={},
        )

        with pytest.raises(ValueError, match="contains blocked term 'secret'"):
            middleware.before_request(request)

    def test_before_request_masks_blocked_term(self):
        """Masks blocked term when mode is 'mask'."""
        middleware = PromptShieldMiddleware(
            denied_terms=["secret"],
            mask="[HIDDEN]",
            on_violation="mask",
        )
        request = LLMRequest(
            system_prompt="System",
            user_prompt="Tell me the secret",
            metadata={},
        )

        result = middleware.before_request(request)

        assert "[HIDDEN]" in result.user_prompt
        assert "secret" not in result.user_prompt

    def test_before_request_logs_blocked_term(self, caplog):
        """Logs when blocked term is found in log mode."""
        middleware = PromptShieldMiddleware(
            denied_terms=["secret"],
            on_violation="log",
        )
        request = LLMRequest(
            system_prompt="System",
            user_prompt="This is a secret message",
            metadata={},
        )

        with caplog.at_level(logging.WARNING):
            result = middleware.before_request(request)

        assert result is request  # Original returned unchanged
        assert "blocked term" in caplog.text.lower()

    def test_case_insensitive_matching(self):
        """Blocked terms are matched case-insensitively."""
        middleware = PromptShieldMiddleware(
            denied_terms=["SECRET"],
            on_violation="abort",
        )
        request = LLMRequest(
            system_prompt="System",
            user_prompt="Tell me the secret",
            metadata={},
        )

        with pytest.raises(ValueError, match="contains blocked term"):
            middleware.before_request(request)

    def test_invalid_mode_defaults_to_abort(self):
        """Invalid on_violation mode defaults to 'abort'."""
        middleware = PromptShieldMiddleware(on_violation="invalid")
        assert middleware.mode == "abort"

    def test_empty_term_is_ignored(self):
        """Empty strings in denied_terms are ignored."""
        middleware = PromptShieldMiddleware(
            denied_terms=["", "secret"],
            on_violation="abort",
        )
        request = LLMRequest(
            system_prompt="System",
            user_prompt="Just a normal message",  # Contains empty string but shouldn't match
            metadata={},
        )

        # Should not raise because empty term is skipped
        result = middleware.before_request(request)
        assert result is request


class TestHealthMonitorMiddleware:
    """Tests for HealthMonitorMiddleware."""

    def test_init_defaults(self):
        """Initializes with default values."""
        middleware = HealthMonitorMiddleware()

        assert middleware.interval == 60.0
        assert middleware.window == 50
        assert middleware.channel == "dmp.health"
        assert middleware.include_latency is True

    def test_init_custom_options(self):
        """Accepts custom options."""
        middleware = HealthMonitorMiddleware(
            heartbeat_interval=30.0,
            stats_window=100,
            channel="custom.health",
            include_latency=False,
        )

        assert middleware.interval == 30.0
        assert middleware.window == 100
        assert middleware.include_latency is False

    def test_init_rejects_negative_interval(self):
        """Rejects negative heartbeat interval."""
        with pytest.raises(ValueError, match="non-negative"):
            HealthMonitorMiddleware(heartbeat_interval=-1)

    def test_before_request_tracks_inflight(self, sample_request):
        """Before request tracks inflight requests."""
        middleware = HealthMonitorMiddleware()

        middleware.before_request(sample_request)

        assert id(sample_request) in middleware._inflight

    def test_after_response_clears_inflight(self, sample_request, sample_response):
        """After response clears inflight tracking."""
        middleware = HealthMonitorMiddleware()
        middleware.before_request(sample_request)

        middleware.after_response(sample_request, sample_response)

        assert id(sample_request) not in middleware._inflight

    def test_tracks_request_counts(self, sample_request, sample_response):
        """Tracks total request counts."""
        middleware = HealthMonitorMiddleware(heartbeat_interval=0)  # Emit immediately
        middleware.before_request(sample_request)

        middleware.after_response(sample_request, sample_response)

        assert middleware._total_requests == 1

    def test_tracks_failure_counts(self, sample_request):
        """Tracks failure counts from error responses."""
        middleware = HealthMonitorMiddleware(heartbeat_interval=0)
        error_response = {"error": "API Error", "content": None}

        middleware.before_request(sample_request)
        middleware.after_response(sample_request, error_response)

        assert middleware._total_failures == 1

    def test_tracks_latencies(self, sample_request, sample_response):
        """Tracks latencies when include_latency is True."""
        middleware = HealthMonitorMiddleware(
            heartbeat_interval=0,
            include_latency=True,
        )
        middleware.before_request(sample_request)
        time.sleep(0.01)  # Ensure measurable latency

        middleware.after_response(sample_request, sample_response)

        assert len(middleware._latencies) == 1
        assert middleware._latencies[0] >= 0.01

    def test_emits_heartbeat(self, sample_request, sample_response, caplog):
        """Emits heartbeat at interval."""
        middleware = HealthMonitorMiddleware(heartbeat_interval=0)  # Emit immediately

        with caplog.at_level(logging.INFO):
            middleware.before_request(sample_request)
            middleware.after_response(sample_request, sample_response)

        assert "health heartbeat" in caplog.text

    def test_window_limits_latencies(self):
        """Stats window limits stored latencies."""
        middleware = HealthMonitorMiddleware(
            stats_window=3,
            heartbeat_interval=999,  # Don't emit
        )

        for i in range(5):
            request = LLMRequest(
                system_prompt="S",
                user_prompt=f"U{i}",
                metadata={},
            )
            middleware.before_request(request)
            middleware.after_response(request, {"content": "ok"})

        assert len(middleware._latencies) == 3


class TestAzureContentSafetyMiddleware:
    """Tests for AzureContentSafetyMiddleware."""

    def test_init_requires_endpoint(self):
        """Requires endpoint."""
        with pytest.raises(ValueError, match="requires an endpoint"):
            AzureContentSafetyMiddleware(endpoint="", key="test-key")

    def test_init_requires_key(self):
        """Requires API key or key_env."""
        with pytest.raises(ValueError, match="requires an API key"):
            AzureContentSafetyMiddleware(endpoint="https://example.com")

    def test_init_key_from_env(self, monkeypatch):
        """API key resolved from environment."""
        monkeypatch.setenv("MY_CONTENT_SAFETY_KEY", "env-key")

        middleware = AzureContentSafetyMiddleware(
            endpoint="https://example.com",
            key_env="MY_CONTENT_SAFETY_KEY",
        )

        assert middleware.key == "env-key"

    def test_init_defaults(self):
        """Initializes with default values."""
        middleware = AzureContentSafetyMiddleware(
            endpoint="https://example.com",
            key="test-key",
        )

        assert middleware.api_version == "2023-10-01"
        assert middleware.categories == ["Hate", "Violence", "SelfHarm", "Sexual"]
        assert middleware.threshold == 4
        assert middleware.mode == "abort"
        assert middleware.on_error == "abort"

    def test_init_custom_options(self):
        """Accepts custom options."""
        middleware = AzureContentSafetyMiddleware(
            endpoint="https://custom.com/",
            key="test-key",
            api_version="2024-01-01",
            categories=["Hate"],
            severity_threshold=2,
            on_violation="mask",
            mask="[BLOCKED]",
            on_error="skip",
        )

        assert middleware.endpoint == "https://custom.com"  # Trailing slash stripped
        assert middleware.api_version == "2024-01-01"
        assert middleware.categories == ["Hate"]
        assert middleware.threshold == 2
        assert middleware.mode == "mask"
        assert middleware.mask == "[BLOCKED]"
        assert middleware.on_error == "skip"

    @patch("requests.post")
    def test_before_request_passes_safe_content(self, mock_post, sample_request):
        """Safe content passes through."""
        mock_post.return_value.json.return_value = {
            "results": [{"category": "Hate", "severity": 0}],
        }
        mock_post.return_value.raise_for_status = MagicMock()

        middleware = AzureContentSafetyMiddleware(
            endpoint="https://example.com",
            key="test-key",
        )
        result = middleware.before_request(sample_request)

        assert result is sample_request

    @patch("requests.post")
    def test_before_request_aborts_flagged_content(self, mock_post, sample_request):
        """Flagged content causes abort."""
        mock_post.return_value.json.return_value = {
            "results": [{"category": "Violence", "severity": 6}],
        }
        mock_post.return_value.raise_for_status = MagicMock()

        middleware = AzureContentSafetyMiddleware(
            endpoint="https://example.com",
            key="test-key",
            severity_threshold=4,
            on_violation="abort",
        )

        with pytest.raises(ValueError, match="blocked by Azure Content Safety"):
            middleware.before_request(sample_request)

    @patch("requests.post")
    def test_before_request_masks_flagged_content(self, mock_post, sample_request):
        """Flagged content is masked when mode is 'mask'."""
        mock_post.return_value.json.return_value = {
            "results": [{"category": "Violence", "severity": 6}],
        }
        mock_post.return_value.raise_for_status = MagicMock()

        middleware = AzureContentSafetyMiddleware(
            endpoint="https://example.com",
            key="test-key",
            severity_threshold=4,
            on_violation="mask",
            mask="[REMOVED]",
        )
        result = middleware.before_request(sample_request)

        assert result.user_prompt == "[REMOVED]"

    @patch("requests.post")
    def test_before_request_logs_flagged_content(self, mock_post, sample_request, caplog):
        """Flagged content is logged when mode is 'log'."""
        mock_post.return_value.json.return_value = {
            "results": [{"category": "Violence", "severity": 6}],
        }
        mock_post.return_value.raise_for_status = MagicMock()

        middleware = AzureContentSafetyMiddleware(
            endpoint="https://example.com",
            key="test-key",
            on_violation="log",
        )

        with caplog.at_level(logging.WARNING):
            result = middleware.before_request(sample_request)

        assert result is sample_request
        assert "flagged" in caplog.text.lower()

    @patch("requests.post")
    def test_analyze_text_constructs_correct_request(self, mock_post, sample_request):
        """Constructs correct API request."""
        mock_post.return_value.json.return_value = {"results": []}
        mock_post.return_value.raise_for_status = MagicMock()

        middleware = AzureContentSafetyMiddleware(
            endpoint="https://my-endpoint.com",
            key="my-api-key",
            api_version="2024-01-01",
            categories=["Hate", "Violence"],
        )
        middleware.before_request(sample_request)

        call_args = mock_post.call_args
        assert "my-endpoint.com/contentsafety/text:analyze" in call_args[0][0]
        assert "api-version=2024-01-01" in call_args[0][0]
        assert call_args[1]["headers"]["Ocp-Apim-Subscription-Key"] == "my-api-key"
        assert call_args[1]["json"]["categories"] == ["Hate", "Violence"]


class TestTracingMiddleware:
    """Tests for TracingMiddleware."""

    def test_init_defaults(self):
        """Initializes with default values."""
        middleware = TracingMiddleware()

        assert middleware.sink == "log"
        assert middleware.sample_rate == 1.0
        assert middleware.include_prompts is False
        assert middleware.include_response is False
        assert middleware.max_chars == 8000
        assert middleware.channel == "dmp.trace"

    def test_init_custom_options(self, tmp_path):
        """Accepts custom options."""
        trace_path = tmp_path / "traces.jsonl"
        middleware = TracingMiddleware(
            sink="jsonl",
            path=str(trace_path),
            sample_rate=0.5,
            include_prompts=True,
            include_response=True,
            max_chars=1000,
            redact_fields=["password", "api_key"],
            redact_value="***",
            channel="custom.trace",
        )

        assert middleware.sink == "jsonl"
        assert middleware.path == trace_path
        assert middleware.sample_rate == 0.5
        assert middleware.include_prompts is True
        assert middleware.include_response is True
        assert middleware.max_chars == 1000
        assert middleware.redact_fields == {"password", "api_key"}
        assert middleware.redact_value == "***"

    def test_init_rejects_invalid_sink(self):
        """Rejects invalid sink type."""
        with pytest.raises(ValueError, match="must be 'log', 'jsonl', or 'otel'"):
            TracingMiddleware(sink="invalid")

    def test_init_rejects_invalid_sample_rate(self):
        """Rejects sample rate outside 0-1 range."""
        with pytest.raises(ValueError, match="sample_rate must be between 0 and 1"):
            TracingMiddleware(sample_rate=1.5)

        with pytest.raises(ValueError, match="sample_rate must be between 0 and 1"):
            TracingMiddleware(sample_rate=-0.1)

    def test_before_request_log_sink(self, sample_request, caplog):
        """Before request logs with log sink."""
        middleware = TracingMiddleware(sink="log", sample_rate=1.0)

        with caplog.at_level(logging.INFO):
            result = middleware.before_request(sample_request)

        assert result is sample_request
        assert "llm_start" in caplog.text

    def test_before_request_includes_prompts_when_enabled(self, sample_request, caplog):
        """Before request logs prompts when enabled."""
        middleware = TracingMiddleware(
            sink="log",
            include_prompts=True,
            sample_rate=1.0,
        )

        with caplog.at_level(logging.INFO):
            middleware.before_request(sample_request)

        assert "You are a helpful assistant" in caplog.text
        assert "What is 2 + 2?" in caplog.text

    def test_after_response_log_sink(self, sample_request, sample_response, caplog):
        """After response logs with log sink."""
        middleware = TracingMiddleware(sink="log", sample_rate=1.0)
        middleware.before_request(sample_request)

        with caplog.at_level(logging.INFO):
            result = middleware.after_response(sample_request, sample_response)

        assert result is sample_response
        assert "llm_end" in caplog.text
        assert "latency_ms" in caplog.text

    def test_after_response_includes_metrics(self, sample_request, sample_response, caplog):
        """After response includes token metrics."""
        middleware = TracingMiddleware(sink="log", sample_rate=1.0)
        middleware.before_request(sample_request)

        with caplog.at_level(logging.INFO):
            middleware.after_response(sample_request, sample_response)

        assert "tokens_prompt" in caplog.text
        assert "20" in caplog.text

    def test_jsonl_sink_writes_to_file(self, tmp_path, sample_request, sample_response):
        """JSONL sink writes traces to file."""
        trace_path = tmp_path / "traces.jsonl"
        middleware = TracingMiddleware(
            sink="jsonl",
            path=str(trace_path),
            sample_rate=1.0,
        )

        middleware.before_request(sample_request)
        middleware.after_response(sample_request, sample_response)

        assert trace_path.exists()
        lines = trace_path.read_text().strip().split("\n")
        assert len(lines) == 2  # start and end events

    def test_sampling_skips_some_requests(self, sample_request, sample_response):
        """Sampling skips some requests."""
        middleware = TracingMiddleware(sink="log", sample_rate=0.0)  # 0% sampling

        # With 0% sample rate, nothing should be tracked
        middleware.before_request(sample_request)

        assert id(sample_request) not in middleware._inflight

    def test_truncates_long_prompts(self, sample_request, caplog):
        """Truncates prompts exceeding max_chars."""
        long_request = LLMRequest(
            system_prompt="A" * 200,
            user_prompt="B" * 200,
            metadata={},
        )
        middleware = TracingMiddleware(
            sink="log",
            include_prompts=True,
            max_chars=50,
            sample_rate=1.0,
        )

        with caplog.at_level(logging.INFO):
            middleware.before_request(long_request)

        assert "[truncated]" in caplog.text

    def test_redacts_sensitive_fields(self, caplog):
        """Redacts sensitive fields from metadata."""
        request = LLMRequest(
            system_prompt="System",
            user_prompt="User",
            metadata={"password": "secret123", "user_id": "123"},
        )
        middleware = TracingMiddleware(
            sink="log",
            redact_fields=["password"],
            redact_value="[HIDDEN]",
            sample_rate=1.0,
        )

        with caplog.at_level(logging.INFO):
            middleware.before_request(request)

        assert "secret123" not in caplog.text
        assert "[HIDDEN]" in caplog.text
        assert "123" in caplog.text  # user_id should not be redacted

    def test_tracks_error_status(self, sample_request, caplog):
        """Tracks error status from response."""
        error_response = {"error": "API failure", "content": None}
        middleware = TracingMiddleware(sink="log", sample_rate=1.0)
        middleware.before_request(sample_request)

        with caplog.at_level(logging.INFO):
            middleware.after_response(sample_request, error_response)

        assert "'status': 'error'" in caplog.text

    def test_includes_response_content_when_enabled(self, sample_request, sample_response, caplog):
        """Includes response content when enabled."""
        middleware = TracingMiddleware(
            sink="log",
            include_response=True,
            sample_rate=1.0,
        )
        middleware.before_request(sample_request)

        with caplog.at_level(logging.INFO):
            middleware.after_response(sample_request, sample_response)

        assert "The answer is 4" in caplog.text

    def test_includes_retry_info(self, sample_request, caplog):
        """Includes retry information when available."""
        response_with_retry = {
            "content": "OK",
            "retry": {"attempts": 2, "max_attempts": 3},
        }
        middleware = TracingMiddleware(sink="log", sample_rate=1.0)
        middleware.before_request(sample_request)

        with caplog.at_level(logging.INFO):
            middleware.after_response(sample_request, response_with_retry)

        assert "attempts_used" in caplog.text
        assert "2" in caplog.text


class TestMiddlewareRegistration:
    """Tests for middleware registration with registry."""

    def test_audit_logger_registered(self):
        """Audit logger is registered."""
        from elspeth.core.llm.registry import create_middleware

        middleware = create_middleware({"name": "audit_logger", "options": {}})
        assert isinstance(middleware, AuditMiddleware)

    def test_prompt_shield_registered(self):
        """Prompt shield is registered."""
        from elspeth.core.llm.registry import create_middleware

        middleware = create_middleware({
            "name": "prompt_shield",
            "options": {"denied_terms": ["test"]},
        })
        assert isinstance(middleware, PromptShieldMiddleware)

    def test_health_monitor_registered(self):
        """Health monitor is registered."""
        from elspeth.core.llm.registry import create_middleware

        middleware = create_middleware({"name": "health_monitor", "options": {}})
        assert isinstance(middleware, HealthMonitorMiddleware)

    def test_azure_content_safety_registered(self):
        """Azure content safety is registered."""
        from elspeth.core.llm.registry import create_middleware

        middleware = create_middleware({
            "name": "azure_content_safety",
            "options": {"endpoint": "https://example.com", "key": "test"},
        })
        assert isinstance(middleware, AzureContentSafetyMiddleware)

    def test_tracing_registered(self):
        """Tracing is registered."""
        from elspeth.core.llm.registry import create_middleware

        middleware = create_middleware({"name": "tracing", "options": {}})
        assert isinstance(middleware, TracingMiddleware)


class TestMiddlewareSchemas:
    """Tests for middleware schema validation."""

    def test_audit_validates_options(self):
        """Audit logger validates options against schema."""
        from elspeth.core.llm.registry import validate_middleware_definition

        # Should not raise with valid options
        validate_middleware_definition({
            "name": "audit_logger",
            "options": {"include_prompts": True, "channel": "custom"},
        })

    def test_prompt_shield_validates_options(self):
        """Prompt shield validates options against schema."""
        from elspeth.core.llm.registry import validate_middleware_definition

        # Should not raise with valid options
        validate_middleware_definition({
            "name": "prompt_shield",
            "options": {"denied_terms": ["secret"], "on_violation": "mask"},
        })

    def test_health_monitor_validates_options(self):
        """Health monitor validates options against schema."""
        from elspeth.core.llm.registry import validate_middleware_definition

        # Should not raise with valid options
        validate_middleware_definition({
            "name": "health_monitor",
            "options": {"heartbeat_interval": 30.0, "stats_window": 100},
        })

    def test_tracing_validates_options(self):
        """Tracing validates options against schema."""
        from elspeth.core.llm.registry import validate_middleware_definition

        # Should not raise with valid options
        validate_middleware_definition({
            "name": "tracing",
            "options": {"sink": "jsonl", "sample_rate": 0.5},
        })

    def test_unknown_middleware_raises(self):
        """Unknown middleware name raises error."""
        from elspeth.core.llm.registry import validate_middleware_definition
        from elspeth.core.validation import ConfigurationError

        with pytest.raises(ConfigurationError, match="Unknown LLM middleware"):
            validate_middleware_definition({"name": "nonexistent"})

    def test_empty_definition_raises(self):
        """Empty definition raises error."""
        from elspeth.core.llm.registry import validate_middleware_definition
        from elspeth.core.validation import ConfigurationError

        with pytest.raises(ConfigurationError, match="cannot be empty"):
            validate_middleware_definition({})

    def test_missing_name_raises(self):
        """Missing name raises error."""
        from elspeth.core.llm.registry import validate_middleware_definition
        from elspeth.core.validation import ConfigurationError

        with pytest.raises(ConfigurationError, match="missing 'name'"):
            validate_middleware_definition({"options": {}})
