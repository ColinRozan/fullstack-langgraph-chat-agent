"""Retry and circuit-breaker configuration using tenacity.

Provides decorators that can be applied to any LLM call or external service
invocation to gain exponential back-off and optional circuit-breaking.
"""

from __future__ import annotations

import functools
import os
import threading
import time
from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from agent.observability import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
DEFAULT_TIMEOUT = int(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))

# ---------------------------------------------------------------------------
# Simple in-memory circuit breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Thread-safe circuit breaker with half-open support.

    States:
        *CLOSED*   – requests flow through normally.
        *OPEN*     – requests are rejected immediately.
        *HALF_OPEN* – one probe request is allowed to test recovery.
    """

    def __init__(
        self,
        failure_threshold: int = 10,
        recovery_timeout: float = 30.0,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name

        self._failures = 0
        self._last_failure_time: float = 0.0
        self._state = "closed"
        self._lock = threading.RLock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if self._state == "open":
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = "half_open"
                    logger.info("circuit_breaker_half_open", name=self.name)
                else:
                    raise RuntimeError(
                        f"Circuit breaker '{self.name}' is OPEN. Call rejected."
                    )

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self._record_failure()
            raise

        self._record_success()
        return result

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self.failure_threshold:
                if self._state != "open":
                    logger.warning(
                        "circuit_breaker_opened",
                        name=self.name,
                        failures=self._failures,
                    )
                self._state = "open"

    def _record_success(self) -> None:
        with self._lock:
            if self._state == "half_open":
                logger.info("circuit_breaker_closed", name=self.name)
            self._state = "closed"
            self._failures = 0


# Global breakers registry (name -> CircuitBreaker)
_BREAKERS: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Return (or create) a circuit breaker identified by *name*."""
    if name not in _BREAKERS:
        _BREAKERS[name] = CircuitBreaker(name=name)
    return _BREAKERS[name]


# ---------------------------------------------------------------------------
# Retry decorators
# ---------------------------------------------------------------------------


def retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator: exponential back-off retry.

    Waits 1s, 2s, 4s … up to 30s between attempts.
    """
    return retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, "warning"),
        reraise=True,
    )


def with_circuit_breaker(name: str) -> Callable[[F], F]:
    """Decorator: wrap the function in a named circuit breaker."""

    def decorator(fn: F) -> F:
        breaker = get_circuit_breaker(name)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return breaker.call(fn, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
