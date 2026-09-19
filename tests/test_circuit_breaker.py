from __future__ import annotations

from llm_gateway.config import CircuitBreakerConfig
from llm_gateway.routing.circuit_breaker import CircuitBreaker, CircuitState
from llm_gateway.telemetry import metrics


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def breaker(clock: FakeClock, **cfg: int | float) -> CircuitBreaker:
    config = CircuitBreakerConfig(
        failure_threshold=3, recovery_timeout_s=10, half_open_max_requests=1, **cfg
    )
    return CircuitBreaker("cb-test", config, clock=clock)


def test_opens_after_consecutive_failures_and_success_resets_count() -> None:
    clock = FakeClock()
    cb = breaker(clock)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()  # resets the consecutive counter
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.CLOSED
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    assert not cb.allow_request()
    assert metrics.CIRCUIT_STATE.labels(provider="cb-test")._value.get() == 1


def test_half_open_admits_limited_probes_then_closes() -> None:
    clock = FakeClock()
    cb = breaker(clock)
    for _ in range(3):
        cb.record_failure()
    clock.now += 9.9
    assert cb.state is CircuitState.OPEN
    clock.now += 0.2
    assert cb.state is CircuitState.HALF_OPEN
    assert cb.allow_request()  # the probe
    assert not cb.allow_request()  # only one probe at a time
    cb.record_success()
    assert cb.state is CircuitState.CLOSED
    assert cb.allow_request()


def test_half_open_failure_reopens() -> None:
    clock = FakeClock()
    cb = breaker(clock)
    for _ in range(3):
        cb.record_failure()
    clock.now += 11
    assert cb.allow_request()
    cb.record_failure("boom")
    assert cb.state is CircuitState.OPEN
    snap = cb.snapshot()
    assert snap["state"] == "open"
    assert snap["last_failure"] == "boom"
    assert 9.9 < snap["retry_in_s"] <= 10


def test_success_threshold_and_release() -> None:
    clock = FakeClock()
    cb = breaker(clock, success_threshold=2)
    for _ in range(3):
        cb.record_failure()
    clock.now += 11
    assert cb.allow_request()
    cb.release()  # neutral outcome gives the probe slot back
    assert cb.allow_request()
    cb.record_success()
    assert cb.state is CircuitState.HALF_OPEN
    assert cb.allow_request()
    cb.record_success()
    assert cb.state is CircuitState.CLOSED


def test_manual_reset() -> None:
    clock = FakeClock()
    cb = breaker(clock)
    for _ in range(3):
        cb.record_failure()
    cb.reset()
    assert cb.state is CircuitState.CLOSED
    assert cb.allow_request()
