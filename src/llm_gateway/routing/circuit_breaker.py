"""Per-provider circuit breaker (closed -> open -> half-open -> closed).

* **closed**: requests flow; consecutive failures are counted. Reaching
  ``failure_threshold`` opens the circuit.
* **open**: requests are rejected immediately (the router skips to the next target)
  until ``recovery_timeout_s`` has elapsed.
* **half-open**: up to ``half_open_max_requests`` probe requests are let through.
  ``success_threshold`` successes close the circuit; any failure re-opens it.

State is per gateway replica (in-process). That is deliberate: it reacts in
microseconds, needs no coordination, and each replica independently discovers a bad
upstream after a handful of failures.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import IntEnum
from typing import Any

from llm_gateway.config import CircuitBreakerConfig
from llm_gateway.telemetry import metrics


class CircuitState(IntEnum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2

    @property
    def label(self) -> str:
        return self.name.lower()


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._half_open_in_flight = 0
        self._half_open_successes = 0
        self._opened_at: float | None = None
        self._last_failure: str | None = None
        metrics.CIRCUIT_STATE.labels(provider=name).set(CircuitState.CLOSED)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def allow_request(self) -> bool:
        """Reserve permission for one call. Must be followed by a ``record_*`` call."""
        with self._lock:
            self._maybe_half_open()
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.OPEN:
                return False
            if self._half_open_in_flight < self.config.half_open_max_requests:
                self._half_open_in_flight += 1
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition(CircuitState.CLOSED)
            else:
                self._failures = 0

    def record_failure(self, reason: str | None = None) -> None:
        with self._lock:
            self._last_failure = reason
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._transition(CircuitState.OPEN)
                return
            if self._state is CircuitState.CLOSED:
                self._failures += 1
                if self._failures >= self.config.failure_threshold:
                    self._transition(CircuitState.OPEN)

    def release(self) -> None:
        """Give back a half-open reservation without judging provider health."""
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._transition(CircuitState.CLOSED)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._maybe_half_open()
            retry_in = None
            if self._state is CircuitState.OPEN and self._opened_at is not None:
                retry_in = max(
                    0.0, self._opened_at + self.config.recovery_timeout_s - self._clock()
                )
            return {
                "state": self._state.label,
                "consecutive_failures": self._failures,
                "last_failure": self._last_failure,
                "retry_in_s": None if retry_in is None else round(retry_in, 3),
            }

    # -- internals (lock held) -------------------------------------------------------------

    def _maybe_half_open(self) -> None:
        if (
            self._state is CircuitState.OPEN
            and self._opened_at is not None
            and self._clock() - self._opened_at >= self.config.recovery_timeout_s
        ):
            self._transition(CircuitState.HALF_OPEN)

    def _transition(self, state: CircuitState) -> None:
        if state is self._state and state is not CircuitState.OPEN:
            return
        self._state = state
        self._failures = 0
        self._half_open_in_flight = 0
        self._half_open_successes = 0
        self._opened_at = self._clock() if state is CircuitState.OPEN else None
        metrics.CIRCUIT_STATE.labels(provider=self.name).set(state)
        metrics.CIRCUIT_TRANSITIONS.labels(provider=self.name, to_state=state.label).inc()


class CircuitBreakerRegistry:
    def __init__(
        self, config: CircuitBreakerConfig, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.config = config
        self._clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, provider: str) -> CircuitBreaker:
        breaker = self._breakers.get(provider)
        if breaker is None:
            breaker = CircuitBreaker(provider, self.config, clock=self._clock)
            self._breakers[provider] = breaker
        return breaker

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {name: breaker.snapshot() for name, breaker in sorted(self._breakers.items())}
