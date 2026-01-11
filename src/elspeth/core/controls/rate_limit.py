"""Rate limiter protocols and implementations."""

from __future__ import annotations

import time
from collections import deque
from contextlib import AbstractContextManager, contextmanager
from threading import Lock
from typing import Any


class RateLimiter:
    """Base protocol for rate limiters."""

    def acquire(self, metadata: dict[str, object] | None = None) -> AbstractContextManager[None]:
        raise NotImplementedError

    def utilization(self) -> float:  # pragma: no cover - default no usage
        return 0.0

    def update_usage(
        self, response: dict[str, Any], metadata: dict[str, object] | None = None
    ) -> None:  # pragma: no cover - optional override
        return


class NoopRateLimiter(RateLimiter):
    def acquire(self, metadata: dict[str, object] | None = None) -> AbstractContextManager[None]:  # pragma: no cover - trivial
        @contextmanager
        def _cm():
            yield

        return _cm()

    def utilization(self) -> float:  # pragma: no cover - trivial
        return 0.0


class FixedWindowRateLimiter(RateLimiter):
    """Simple fixed window limiter enforcing N requests per interval."""

    def __init__(self, requests: int = 1, per_seconds: float = 1.0):
        if requests <= 0 or per_seconds <= 0:
            raise ValueError("requests and per_seconds must be positive")
        self.requests = requests
        self.per_seconds = per_seconds
        self._lock = Lock()
        self._window_start = 0.0
        self._count = 0
        self._usage_ratio = 0.0

    def acquire(self, metadata: dict[str, object] | None = None) -> AbstractContextManager[None]:
        @contextmanager
        def _cm():
            while True:
                with self._lock:
                    now = time.time()
                    elapsed = now - self._window_start
                    if elapsed >= self.per_seconds:
                        self._window_start = now
                        self._count = 0
                    if self._count < self.requests:
                        self._count += 1
                        self._usage_ratio = min(1.0, self._count / self.requests)
                        break
                    sleep_for = self.per_seconds - elapsed
                time.sleep(max(sleep_for, 0.001))
            try:
                yield
            finally:
                pass

        return _cm()

    def utilization(self) -> float:
        with self._lock:
            return min(1.0, self._usage_ratio)


class AdaptiveRateLimiter(RateLimiter):
    """Limiter supporting request and token ceilings per interval.

    Also enforces minimum spacing between requests to prevent burst scenarios.
    """

    def __init__(
        self,
        *,
        requests_per_minute: int,
        tokens_per_minute: int | None = None,
        interval_seconds: float = 60.0,
        min_interval: float | None = None,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.interval = interval_seconds
        # Default min_interval: spread requests evenly over the interval
        # e.g., 200 RPM over 60s = 0.3s between requests
        self.min_interval = min_interval if min_interval is not None else (interval_seconds / requests_per_minute)
        self._lock = Lock()
        self._request_times: deque[float] = deque()
        self._token_records: deque[tuple[float, float]] = deque()
        self._last_utilization = 0.0
        self._last_request_time: float = 0.0

    def acquire(self, metadata: dict[str, object] | None = None) -> AbstractContextManager[None]:
        estimated_tokens = 0.0
        if metadata:
            value = metadata.get("estimated_tokens") or metadata.get("expected_tokens")
            if isinstance(value, (int, float, str, bytes)):
                try:
                    estimated_tokens = float(value)
                except (TypeError, ValueError):
                    estimated_tokens = 0.0

        @contextmanager
        def _cm():
            # First, enforce minimum interval (burst prevention)
            # This must be done atomically to prevent race conditions
            with self._lock:
                now = time.time()
                time_since_last = now - self._last_request_time
                if time_since_last < self.min_interval:
                    sleep_for = self.min_interval - time_since_last
                else:
                    sleep_for = 0.0
                # Reserve our slot by updating _last_request_time NOW
                # This prevents other threads from also thinking they can go immediately
                self._last_request_time = now + sleep_for

            # Sleep outside the lock if needed
            if sleep_for > 0:
                time.sleep(sleep_for)

            # Now do the per-minute rate limit check
            while True:
                with self._lock:
                    now = time.time()
                    self._trim(now)
                    request_usage = len(self._request_times) / self.requests_per_minute
                    token_usage = 0.0
                    if self.tokens_per_minute:
                        current_tokens = sum(tokens for _, tokens in self._token_records)
                        token_usage = (current_tokens + estimated_tokens) / self.tokens_per_minute
                    self._last_utilization = max(request_usage, token_usage)
                    if request_usage < 1.0 and (self.tokens_per_minute is None or token_usage < 1.0):
                        self._request_times.append(now)
                        break
                    sleep_for = self._next_available_time(now)
                time.sleep(sleep_for)
            try:
                yield
            finally:
                pass

        return _cm()

    def update_usage(self, response: dict[str, Any], metadata: dict[str, object] | None = None) -> None:
        if self.tokens_per_minute is None:
            return
        metrics = response.get("metrics") if isinstance(response, dict) else None
        tokens = 0.0
        if isinstance(metrics, dict):
            tokens += float(metrics.get("prompt_tokens") or 0)
            tokens += float(metrics.get("completion_tokens") or 0)
        if tokens <= 0:
            return
        with self._lock:
            now = time.time()
            self._trim(now)
            self._token_records.append((now, tokens))

    def utilization(self) -> float:
        with self._lock:
            self._trim(time.time())
            request_usage = len(self._request_times) / self.requests_per_minute
            token_usage = 0.0
            if self.tokens_per_minute:
                current_tokens = sum(tokens for _, tokens in self._token_records)
                token_usage = current_tokens / self.tokens_per_minute
            self._last_utilization = max(request_usage, token_usage)
            return min(1.0, self._last_utilization)

    def _trim(self, now: float) -> None:
        cutoff = now - self.interval
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()
        while self._token_records and self._token_records[0][0] < cutoff:
            self._token_records.popleft()

    def _next_available_time(self, now: float) -> float:
        next_times = []
        if self._request_times:
            next_times.append(max(0.0, self._request_times[0] + self.interval - now))
        if self._token_records:
            next_times.append(max(0.0, self._token_records[0][0] + self.interval - now))
        return min(next_times) if next_times else 0.1


__all__ = [
    "AdaptiveRateLimiter",
    "FixedWindowRateLimiter",
    "NoopRateLimiter",
    "RateLimiter",
]
