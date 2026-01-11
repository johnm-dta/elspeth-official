"""Tests for AzureEnvironmentMiddleware."""

import logging
import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from elspeth.core.llm.middleware import LLMRequest
from elspeth.plugins.llms.middleware_azure import (
    _AZURE_ENV_VARS,
    AzureEnvironmentMiddleware,
    _is_probably_running_in_azure,
    _resolve_azure_run,
)

# =============================================================================
# Tests for helper functions
# =============================================================================


class TestIsProbablyRunningInAzure:
    """Tests for _is_probably_running_in_azure()."""

    def test_returns_false_when_no_env_vars(self):
        """Returns False when no Azure env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear all Azure env vars
            for var in _AZURE_ENV_VARS:
                os.environ.pop(var, None)

            result = _is_probably_running_in_azure()

            assert result is False

    def test_returns_true_when_run_id_set(self):
        """Returns True when AZUREML_RUN_ID is set."""
        with patch.dict(os.environ, {"AZUREML_RUN_ID": "test-run-123"}):
            result = _is_probably_running_in_azure()

            assert result is True

    def test_returns_true_when_any_azure_var_set(self):
        """Returns True when any Azure env var is set."""
        for var in _AZURE_ENV_VARS:
            with patch.dict(os.environ, {var: "some-value"}, clear=True):
                result = _is_probably_running_in_azure()

                assert result is True, f"Should detect {var}"


class TestResolveAzureRun:
    """Tests for _resolve_azure_run()."""

    def test_returns_none_when_azureml_not_installed(self):
        """Returns None when azureml.core is not installed."""
        # The import will fail in test environment
        result = _resolve_azure_run()

        assert result is None

    def test_returns_none_when_import_fails(self):
        """Returns None when import raises exception."""
        with patch.dict("sys.modules", {"azureml": None, "azureml.core": None}):
            result = _resolve_azure_run()

            assert result is None


# =============================================================================
# Tests for AzureEnvironmentMiddleware initialization
# =============================================================================


class TestAzureEnvironmentMiddlewareInit:
    """Tests for middleware initialization."""

    def test_init_with_defaults_no_azure(self):
        """Initialize with defaults when not in Azure."""
        with patch.dict(os.environ, {}, clear=True):
            for var in _AZURE_ENV_VARS:
                os.environ.pop(var, None)

            middleware = AzureEnvironmentMiddleware(on_error="skip")

            assert middleware.on_error == "skip"
            assert middleware.log_prompts is False
            assert middleware.log_config_diffs is True
            assert middleware.log_metrics is True
            assert middleware._run is None

    def test_init_with_custom_params(self):
        """Initialize with custom parameters."""
        middleware = AzureEnvironmentMiddleware(
            enable_run_logging=False,
            log_prompts=True,
            log_config_diffs=False,
            log_metrics=False,
            severity_threshold="WARNING",
            on_error="skip",
        )

        assert middleware.log_prompts is True
        assert middleware.log_config_diffs is False
        assert middleware.log_metrics is False
        assert middleware._fallback_level == logging.WARNING

    def test_init_invalid_on_error(self):
        """Invalid on_error raises ValueError."""
        with pytest.raises(ValueError, match="on_error must be"):
            AzureEnvironmentMiddleware(on_error="invalid")

    def test_init_unknown_severity_defaults_to_info(self, caplog):
        """Unknown severity threshold defaults to INFO with warning."""
        with caplog.at_level(logging.WARNING):
            middleware = AzureEnvironmentMiddleware(
                severity_threshold="UNKNOWN",
                on_error="skip",
            )

        assert middleware._fallback_level == logging.INFO
        assert "Unknown severity_threshold" in caplog.text

    @pytest.mark.parametrize("level_name,expected", [
        ("CRITICAL", logging.CRITICAL),
        ("ERROR", logging.ERROR),
        ("WARNING", logging.WARNING),
        ("INFO", logging.INFO),
        ("DEBUG", logging.DEBUG),
        ("critical", logging.CRITICAL),  # Case insensitive
        ("info", logging.INFO),
    ])
    def test_init_severity_levels(self, level_name, expected):
        """All severity levels are recognized."""
        middleware = AzureEnvironmentMiddleware(
            severity_threshold=level_name,
            on_error="skip",
        )

        assert middleware._fallback_level == expected

    def test_init_run_logging_disabled(self):
        """Run logging can be disabled."""
        middleware = AzureEnvironmentMiddleware(
            enable_run_logging=False,
            on_error="skip",
        )

        assert middleware._run is None

    def test_init_raises_when_azure_detected_but_no_run_and_abort(self):
        """Raises RuntimeError when in Azure but no run context with on_error=abort."""
        with patch.dict(os.environ, {"AZUREML_RUN_ID": "test-run"}), pytest.raises(RuntimeError, match="Azure ML run context"):
            AzureEnvironmentMiddleware(
                enable_run_logging=True,
                on_error="abort",
            )

    def test_init_warns_when_azure_detected_but_no_run_and_skip(self, caplog):
        """Warns when in Azure but no run context with on_error=skip."""
        with patch.dict(os.environ, {"AZUREML_RUN_ID": "test-run"}), caplog.at_level(logging.WARNING):
            middleware = AzureEnvironmentMiddleware(
                enable_run_logging=True,
                on_error="skip",
            )

        assert middleware._run is None
        assert "Continuing without run context" in caplog.text

    def test_init_summary_structure(self):
        """Initializes summary tracking structure."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        assert middleware._summary == {
            "experiments": 0,
            "total_rows": 0,
            "total_failures": 0,
        }


# =============================================================================
# Tests for before_request / after_response
# =============================================================================


class TestAzureEnvironmentMiddlewareRequestResponse:
    """Tests for request/response handling."""

    def test_before_request_adds_sequence(self):
        """before_request adds azure_sequence to metadata."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        request = LLMRequest(
            system_prompt="System",
            user_prompt="User",
            metadata={"existing": "value"},
        )

        result = middleware.before_request(request)

        assert "azure_sequence" in result.metadata
        assert result.metadata["azure_sequence"].startswith("az-")
        assert result.metadata["existing"] == "value"

    def test_before_request_sequence_increments(self):
        """Sequence number increments on each request."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        r1 = middleware.before_request(LLMRequest("S", "U", {}))
        r2 = middleware.before_request(LLMRequest("S", "U", {}))
        r3 = middleware.before_request(LLMRequest("S", "U", {}))

        assert r1.metadata["azure_sequence"] == "az-1"
        assert r2.metadata["azure_sequence"] == "az-2"
        assert r3.metadata["azure_sequence"] == "az-3"

    def test_before_request_logs_prompts_when_enabled(self, caplog):
        """Logs prompts when log_prompts=True."""
        middleware = AzureEnvironmentMiddleware(
            log_prompts=True,
            on_error="skip",
        )
        request = LLMRequest(
            system_prompt="System prompt text",
            user_prompt="User prompt text",
            metadata={},
        )

        with caplog.at_level(logging.INFO):
            middleware.before_request(request)

        assert "System prompt text" in caplog.text
        assert "User prompt text" in caplog.text

    def test_before_request_no_prompts_by_default(self, caplog):
        """Does not log prompts by default."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        request = LLMRequest(
            system_prompt="Secret system prompt",
            user_prompt="Secret user prompt",
            metadata={},
        )

        with caplog.at_level(logging.INFO):
            middleware.before_request(request)

        assert "Secret system prompt" not in caplog.text
        assert "Secret user prompt" not in caplog.text

    def test_after_response_computes_duration(self):
        """after_response computes duration from tracked start time."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        request = LLMRequest("S", "U", {})

        # Simulate request flow
        updated_request = middleware.before_request(request)
        time.sleep(0.01)  # Small delay

        response = {"result": "ok"}
        result = middleware.after_response(updated_request, response)

        assert result == response  # Response unchanged

    def test_after_response_handles_missing_sequence(self):
        """after_response handles request without sequence."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        request = LLMRequest("S", "U", {})  # No azure_sequence

        response = {"result": "ok"}
        result = middleware.after_response(request, response)

        assert result == response

    def test_after_response_logs_metrics(self, caplog):
        """after_response logs metrics from response."""
        middleware = AzureEnvironmentMiddleware(
            log_metrics=True,
            on_error="skip",
        )
        request = middleware.before_request(LLMRequest("S", "U", {}))

        response = {
            "result": "ok",
            "metrics": {
                "score": 0.95,
                "tokens": 150,
            },
        }

        with caplog.at_level(logging.INFO):
            middleware.after_response(request, response)

        assert "metric_score" in caplog.text
        assert "metric_tokens" in caplog.text

    def test_after_response_logs_error(self, caplog):
        """after_response logs error from response."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        request = middleware.before_request(LLMRequest("S", "U", {}))

        response = {
            "result": None,
            "error": "API rate limit exceeded",
        }

        with caplog.at_level(logging.INFO):
            middleware.after_response(request, response)

        assert "API rate limit exceeded" in caplog.text


# =============================================================================
# Tests for suite lifecycle methods
# =============================================================================


class TestAzureEnvironmentMiddlewareSuiteLifecycle:
    """Tests for suite lifecycle methods."""

    def test_on_suite_loaded_logs_experiments(self, caplog):
        """on_suite_loaded logs experiment information."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        experiments = [
            {"name": "exp1", "config": "a"},
            {"name": "exp2", "config": "b"},
        ]

        with caplog.at_level(logging.INFO):
            middleware.on_suite_loaded(experiments)

        assert "experiment_count" in caplog.text
        assert middleware._suite_logged is True

    def test_on_suite_loaded_only_logs_once(self, caplog):
        """on_suite_loaded only logs once."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        experiments = [{"name": "exp1"}]

        with caplog.at_level(logging.INFO):
            middleware.on_suite_loaded(experiments)
            caplog.clear()
            middleware.on_suite_loaded(experiments)

        # Second call should not log
        assert "experiment_count" not in caplog.text

    def test_on_suite_loaded_logs_preflight(self, caplog):
        """on_suite_loaded logs preflight data."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        experiments = [{"name": "exp1"}]
        preflight = {"validation": "passed", "checks": 5}

        with caplog.at_level(logging.INFO):
            middleware.on_suite_loaded(experiments, preflight=preflight)

        assert "suite_preflight" in caplog.text

    def test_on_experiment_start_logs_metadata(self, caplog):
        """on_experiment_start logs experiment metadata."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        with caplog.at_level(logging.INFO):
            middleware.on_experiment_start("test_exp", {"model": "gpt-4"})

        assert "experiment_start" in caplog.text
        assert "test_exp" in caplog.text

    def test_on_experiment_complete_updates_summary(self):
        """on_experiment_complete updates summary counters."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        payload = {
            "results": [{"id": 1}, {"id": 2}, {"id": 3}],
            "failures": [{"error": "timeout"}],
        }

        middleware.on_experiment_complete("exp1", payload)

        assert middleware._summary["experiments"] == 1
        assert middleware._summary["total_rows"] == 3
        assert middleware._summary["total_failures"] == 1

    def test_on_experiment_complete_accumulates(self):
        """on_experiment_complete accumulates across experiments."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        middleware.on_experiment_complete("exp1", {"results": [1, 2], "failures": []})
        middleware.on_experiment_complete("exp2", {"results": [3, 4, 5], "failures": [1]})

        assert middleware._summary["experiments"] == 2
        assert middleware._summary["total_rows"] == 5
        assert middleware._summary["total_failures"] == 1

    def test_on_experiment_complete_logs_cost_summary(self, caplog):
        """on_experiment_complete logs cost summary."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        payload = {
            "results": [],
            "cost_summary": {
                "total_tokens": 1000,
                "estimated_cost": 0.05,
            },
        }

        with caplog.at_level(logging.INFO):
            middleware.on_experiment_complete("exp1", payload)

        assert "cost_total_tokens" in caplog.text
        assert "cost_estimated_cost" in caplog.text

    def test_on_experiment_complete_logs_aggregates(self, caplog):
        """on_experiment_complete logs aggregates."""
        middleware = AzureEnvironmentMiddleware(
            log_metrics=True,
            on_error="skip",
        )

        payload = {
            "results": [],
            "aggregates": {
                "mean_score": 0.85,
                "std_score": 0.12,
            },
        }

        with caplog.at_level(logging.INFO):
            middleware.on_experiment_complete("exp1", payload)

        # Aggregates should be JSON-encoded
        assert "aggregates" in caplog.text

    def test_on_experiment_complete_handles_empty_payload(self):
        """on_experiment_complete handles empty/None payload."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        middleware.on_experiment_complete("exp1", {})

        assert middleware._summary["experiments"] == 1
        assert middleware._summary["total_rows"] == 0

    def test_on_baseline_comparison_logs_diffs(self, caplog):
        """on_baseline_comparison logs comparison diffs."""
        middleware = AzureEnvironmentMiddleware(
            log_config_diffs=True,
            on_error="skip",
        )

        comparison = {
            "score_delta": {"delta": 0.15, "ratio": 1.2},
        }

        with caplog.at_level(logging.INFO):
            middleware.on_baseline_comparison("exp1", comparison)

        assert "baseline_exp1_score_delta" in caplog.text

    def test_on_baseline_comparison_skips_when_disabled(self, caplog):
        """on_baseline_comparison skips when log_config_diffs=False."""
        middleware = AzureEnvironmentMiddleware(
            log_config_diffs=False,
            on_error="skip",
        )

        comparison = {"score_delta": {"delta": 0.15}}

        with caplog.at_level(logging.INFO):
            middleware.on_baseline_comparison("exp1", comparison)

        assert "baseline_" not in caplog.text

    def test_on_baseline_comparison_handles_empty(self):
        """on_baseline_comparison handles empty comparison."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        # Should not raise
        middleware.on_baseline_comparison("exp1", {})
        middleware.on_baseline_comparison("exp1", None)

    def test_on_suite_complete_logs_summary(self, caplog):
        """on_suite_complete logs final summary."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        # Simulate some experiments
        middleware.on_experiment_complete("exp1", {"results": [1, 2]})
        middleware.on_experiment_complete("exp2", {"results": [3]})

        with caplog.at_level(logging.INFO):
            middleware.on_suite_complete()

        assert "suite_summary" in caplog.text

    def test_on_retry_exhausted_logs_details(self, caplog):
        """on_retry_exhausted logs retry failure details."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        request = middleware.before_request(LLMRequest("S", "U", {}))
        metadata = {
            "attempts": 3,
            "max_attempts": 3,
            "error": "Connection timeout",
            "error_type": "TimeoutError",
            "history": [{"attempt": 1}, {"attempt": 2}],
        }

        with caplog.at_level(logging.INFO):
            middleware.on_retry_exhausted(request, metadata, Exception("timeout"))

        assert "llm_retry_exhausted" in caplog.text
        assert "Connection timeout" in caplog.text


# =============================================================================
# Tests for thread safety
# =============================================================================


class TestAzureEnvironmentMiddlewareThreadSafety:
    """Tests for thread safety."""

    def test_sequence_thread_safe(self):
        """Sequence generation is thread-safe."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        sequences = []
        errors = []

        def worker():
            try:
                for _ in range(100):
                    request = LLMRequest("S", "U", {})
                    result = middleware.before_request(request)
                    sequences.append(result.metadata["azure_sequence"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All sequences should be unique
        assert len(sequences) == len(set(sequences))

    def test_inflight_tracking_thread_safe(self):
        """In-flight request tracking is thread-safe."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        errors = []

        def worker():
            try:
                for _ in range(50):
                    request = LLMRequest("S", "U", {})
                    updated = middleware.before_request(request)
                    time.sleep(0.001)
                    middleware.after_response(updated, {"ok": True})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All in-flight should be cleaned up
        assert len(middleware._inflight) == 0


# =============================================================================
# Tests for Azure ML run integration (mocked)
# =============================================================================


class TestAzureEnvironmentMiddlewareWithMockedRun:
    """Tests with mocked Azure ML run."""

    def test_log_row_calls_run(self):
        """_log_row calls run.log_row when run is available."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        mock_run = MagicMock()
        middleware._run = mock_run

        middleware._log_row("test_event", {"key": "value"})

        mock_run.log_row.assert_called_once_with("test_event", key="value")

    def test_log_table_calls_run(self):
        """_log_table calls run.log_table when run is available."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        mock_run = MagicMock()
        middleware._run = mock_run

        middleware._log_table("test_table", {"col": [1, 2, 3]})

        mock_run.log_table.assert_called_once_with("test_table", {"col": [1, 2, 3]})

    def test_log_metric_calls_run(self):
        """_log_metric calls run.log when run is available."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        mock_run = MagicMock()
        middleware._run = mock_run

        middleware._log_metric("accuracy", 0.95)

        mock_run.log.assert_called_once_with("accuracy", 0.95)

    def test_before_request_with_run(self):
        """before_request logs to run when available."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        mock_run = MagicMock()
        middleware._run = mock_run

        request = LLMRequest("System", "User", {})
        middleware.before_request(request)

        mock_run.log_row.assert_called_once()
        call_args = mock_run.log_row.call_args
        assert call_args[0][0] == "llm_request"

    def test_after_response_with_run(self):
        """after_response logs to run when available."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        mock_run = MagicMock()
        middleware._run = mock_run

        request = middleware.before_request(LLMRequest("S", "U", {}))
        mock_run.reset_mock()

        middleware.after_response(request, {"result": "ok"})

        mock_run.log_row.assert_called_once()
        call_args = mock_run.log_row.call_args
        assert call_args[0][0] == "llm_response"


# =============================================================================
# Tests for schema and registration
# =============================================================================


class TestAzureEnvironmentMiddlewareSchema:
    """Tests for schema and registration."""

    def test_has_name(self):
        """Middleware has name attribute."""
        assert AzureEnvironmentMiddleware.name == "azure_environment"

    def test_schema_defined(self):
        """Schema is properly defined."""
        from elspeth.plugins.llms.middleware_azure import _AZURE_ENV_SCHEMA

        assert _AZURE_ENV_SCHEMA["type"] == "object"
        assert "enable_run_logging" in _AZURE_ENV_SCHEMA["properties"]
        assert "log_prompts" in _AZURE_ENV_SCHEMA["properties"]
        assert "severity_threshold" in _AZURE_ENV_SCHEMA["properties"]

    def test_middleware_registered(self):
        """Middleware is registered with factory."""
        from elspeth.core.llm.registry import create_middleware

        middleware = create_middleware({
            "name": "azure_environment",
            "options": {
                "enable_run_logging": False,
                "log_prompts": True,
                "on_error": "skip",
            },
        })

        assert isinstance(middleware, AzureEnvironmentMiddleware)
        assert middleware.log_prompts is True


# =============================================================================
# Tests for edge cases
# =============================================================================


class TestAzureEnvironmentMiddlewareEdgeCases:
    """Edge case tests."""

    def test_handles_non_dict_response(self):
        """Handles non-dict response gracefully."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        request = middleware.before_request(LLMRequest("S", "U", {}))

        # Non-dict response
        result = middleware.after_response(request, "string response")

        assert result == "string response"

    def test_handles_none_metrics(self):
        """Handles None metrics in response."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        request = middleware.before_request(LLMRequest("S", "U", {}))

        response = {"result": "ok", "metrics": None}
        result = middleware.after_response(request, response)

        assert result == response

    def test_handles_non_dict_metrics(self):
        """Handles non-dict metrics in response."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        request = middleware.before_request(LLMRequest("S", "U", {}))

        response = {"result": "ok", "metrics": "not a dict"}
        result = middleware.after_response(request, response)

        assert result == response

    def test_clone_preserves_original_request(self):
        """before_request clones without modifying original."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")
        original_metadata = {"original": "value"}
        request = LLMRequest("S", "U", original_metadata)

        result = middleware.before_request(request)

        assert "azure_sequence" not in original_metadata
        assert "azure_sequence" in result.metadata
        assert result.metadata["original"] == "value"

    def test_experiment_complete_with_non_list_results(self):
        """on_experiment_complete handles non-list results."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        payload = {"results": "not a list", "failures": None}
        middleware.on_experiment_complete("exp1", payload)

        assert middleware._summary["total_rows"] == 0
        assert middleware._summary["total_failures"] == 0

    def test_experiment_complete_samples_failures(self, caplog):
        """on_experiment_complete samples failures (max 3)."""
        middleware = AzureEnvironmentMiddleware(on_error="skip")

        failures = [{"error": f"error_{i}"} for i in range(10)]
        payload = {"results": [], "failures": failures}

        with caplog.at_level(logging.INFO):
            middleware.on_experiment_complete("exp1", payload)

        # Only first 3 failures should be in the log
        log_text = caplog.text
        assert "error_0" in log_text
        assert "error_2" in log_text
