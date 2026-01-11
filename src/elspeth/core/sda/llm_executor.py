"""LLM execution with retry logic for SDA."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from elspeth.core.landscape import get_current_landscape

if TYPE_CHECKING:
    from elspeth.core.controls import CostTracker, RateLimiter
    from elspeth.core.interfaces import LLMClientProtocol
    from elspeth.core.llm.middleware import LLMMiddleware

logger = logging.getLogger(__name__)


class LLMExecutor:
    """Executes LLM calls with middleware, retry, rate limiting, and cost tracking."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        middlewares: list[LLMMiddleware],
        retry_config: dict[str, Any] | None,
        rate_limiter: RateLimiter | None,
        cost_tracker: CostTracker | None,
        cycle_name: str | None = None,
    ):
        """
        Initialize LLM executor.

        Args:
            llm_client: LLM client for generation
            middlewares: Middleware chain
            retry_config: Retry configuration (max_attempts, backoff_multiplier, initial_delay)
            rate_limiter: Rate limiter instance
            cost_tracker: Cost tracker instance
            cycle_name: Name of SDA cycle for metadata
        """
        self.llm_client = llm_client
        self.middlewares = middlewares
        self.retry_config = retry_config or {}
        self.rate_limiter = rate_limiter
        self.cost_tracker = cost_tracker
        self.cycle_name = cycle_name

    def execute(
        self,
        user_prompt: str,
        metadata: dict[str, Any],
        system_prompt: str | None = None,
        skip_rate_limit: bool = False,
    ) -> dict[str, Any]:
        """
        Execute LLM call with retry logic.

        Args:
            user_prompt: User prompt text
            metadata: Request metadata
            system_prompt: System prompt text
            skip_rate_limit: Skip rate limiter acquisition (for pre-acquired slots)

        Returns:
            LLM response dictionary

        Raises:
            Exception: If all retry attempts are exhausted
        """
        from elspeth.core.llm.middleware import LLMRequest

        delay = 0.0
        max_attempts = 1
        backoff = 0.0
        if self.retry_config:
            max_attempts = int(self.retry_config.get("max_attempts", 1))
            delay = float(self.retry_config.get("initial_delay", 0.0))
            backoff = float(self.retry_config.get("backoff_multiplier", 1.0))

        attempt = 0
        last_error: Exception | None = None
        attempt_history: list[dict[str, Any]] = []
        last_request: LLMRequest | None = None

        while attempt < max_attempts:
            attempt += 1
            try:
                request = LLMRequest(
                    system_prompt=system_prompt or "",
                    user_prompt=user_prompt,
                    metadata={**metadata, "attempt": attempt},
                )
                last_request = request
                attempt_start = time.time()

                # Apply middleware chain - before_request
                for middleware in self.middlewares:
                    request = middleware.before_request(request)

                # Apply rate limiting (skip if pre-acquired at launch time)
                if self.rate_limiter and not skip_rate_limit:
                    acquire_context = self.rate_limiter.acquire({"experiment": self.cycle_name, **request.metadata})
                else:
                    acquire_context = None

                # Execute LLM call
                if acquire_context:
                    with acquire_context:
                        response = self.llm_client.generate(
                            system_prompt=request.system_prompt,
                            user_prompt=request.user_prompt,
                            metadata=request.metadata,
                        )
                else:
                    response = self.llm_client.generate(
                        system_prompt=request.system_prompt,
                        user_prompt=request.user_prompt,
                        metadata=request.metadata,
                    )

                # Apply middleware chain - after_response
                for middleware in reversed(self.middlewares):
                    response = middleware.after_response(request, response)

                # Track cost
                if self.cost_tracker:
                    cost_metrics = self.cost_tracker.record(response, {"experiment": self.cycle_name, **request.metadata})
                    if cost_metrics:
                        response.setdefault("metrics", {}).update(cost_metrics)

                # Record attempt success
                attempt_record = {
                    "attempt": attempt,
                    "status": "success",
                    "duration": max(time.time() - attempt_start, 0.0),
                }
                attempt_history.append(attempt_record)

                # Add retry metadata
                response.setdefault("metrics", {})["attempts_used"] = attempt
                response.setdefault(
                    "retry",
                    {
                        "attempts": attempt,
                        "max_attempts": max_attempts,
                        "history": attempt_history,
                    },
                )

                # Update rate limiter usage
                if self.rate_limiter:
                    self.rate_limiter.update_usage(response, request.metadata)

                # Log to landscape if active
                self._log_to_landscape(request, response, metadata)

                return response

            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                attempt_record = {
                    "attempt": attempt,
                    "status": "error",
                    "duration": max(time.time() - attempt_start, 0.0),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                }

                if attempt >= max_attempts:
                    attempt_history.append(attempt_record)
                    break

                sleep_for = delay if delay > 0 else 0
                attempt_record["next_delay"] = sleep_for
                attempt_history.append(attempt_record)

                if sleep_for > 0:
                    time.sleep(sleep_for)

                if backoff and backoff > 0:
                    delay = delay * backoff if delay else backoff

        # All retries exhausted
        assert last_error is not None
        if last_request is not None:
            setattr(last_error, "_dmp_retry_history", attempt_history)
            setattr(last_error, "_dmp_retry_attempts", attempt)
            setattr(last_error, "_dmp_retry_max_attempts", max_attempts)
            try:
                self._notify_retry_exhausted(last_request, last_error, attempt_history)
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("Retry exhausted hook raised", exc_info=True)
        raise last_error

    def _notify_retry_exhausted(self, request: Any, error: Exception, history: list[dict[str, Any]]) -> None:
        """
        Notify middleware about retry exhaustion.

        Args:
            request: The LLM request that failed
            error: The exception that was raised
            history: List of retry attempt records
        """
        metadata = {
            "experiment": self.cycle_name,
            "attempts": getattr(error, "_dmp_retry_attempts", len(history)),
            "max_attempts": getattr(error, "_dmp_retry_max_attempts", len(history)),
            "error": str(error),
            "error_type": error.__class__.__name__,
            "history": history,
        }
        logger.warning(
            "LLM request exhausted retries for experiment '%s' after %s attempts: %s",
            self.cycle_name,
            metadata["attempts"],
            metadata["error"],
        )
        for middleware in self.middlewares:
            hook = getattr(middleware, "on_retry_exhausted", None)
            if callable(hook):
                try:
                    hook(request, metadata, error)
                except Exception:  # pragma: no cover - middleware isolation
                    logger.debug(
                        "Middleware %s retry hook failed",
                        getattr(middleware, "name", middleware),
                        exc_info=True,
                    )

    def _log_to_landscape(self, request: Any, response: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Log LLM call to landscape if active.

        Args:
            request: The LLMRequest object
            response: The LLM response dict
            metadata: Original metadata passed to execute()
        """
        landscape = get_current_landscape()
        if landscape is None:
            return

        landscape.log_llm_call(
            call_id=None,  # Auto-generate
            request={
                "system_prompt": request.system_prompt,
                "user_prompt": request.user_prompt,
            },
            response=response,
            metadata=metadata,
        )
