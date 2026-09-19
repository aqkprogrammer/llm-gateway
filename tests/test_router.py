from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from llm_gateway.config import CircuitBreakerConfig, ProviderConfig, parse_gateway_config
from llm_gateway.errors import AllTargetsFailedError, GatewayError, ProviderError
from llm_gateway.providers.base import Provider
from llm_gateway.routing.circuit_breaker import CircuitBreakerRegistry, CircuitState
from llm_gateway.routing.router import Router
from llm_gateway.schemas import ChatCompletionRequest, ChatResult, StreamEvent, Usage


class ScriptedProvider(Provider):
    """Fails with the scripted errors, in order, then succeeds."""

    def __init__(self, name: str, errors: list[ProviderError | None] | None = None) -> None:
        super().__init__(name, ProviderConfig(type="mock"))
        self.errors = list(errors or [])
        self.calls = 0

    def _next(self) -> None:
        self.calls += 1
        if self.errors:
            err = self.errors.pop(0)
            if err is not None:
                raise err

    async def chat(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> ChatResult:
        self._next()
        return ChatResult(
            model=model, content=f"from {self.name}", finish_reason="stop", usage=Usage(1, 1)
        )

    async def stream(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> AsyncIterator[StreamEvent]:
        self._next()
        yield StreamEvent(content=f"from {self.name}")
        yield StreamEvent(finish_reason="stop")


def err(
    provider: str,
    status: int | None = 503,
    kind: str = "server_error",
    retry_after: float | None = None,
) -> ProviderError:
    return ProviderError(
        "boom", provider=provider, status_code=status, kind=kind, retry_after=retry_after
    )


CONFIG = """
providers:
  a: {type: mock}
  b: {type: mock}
  c: {type: mock}
  nokey: {type: openai, api_key: ""}
routes:
  chain:
    targets: [a:m1, b:m2, c:m3]
    max_retries: 2
    backoff_base_s: 1
    backoff_max_s: 8
  skip-disabled:
    targets: [nokey:gpt, c:m3]
"""


@pytest.fixture
def sleeps() -> list[float]:
    return []


def make_router(providers: dict[str, Provider], sleeps: list[float], threshold: int = 5) -> Router:
    config = parse_gateway_config(CONFIG, environ={})
    breakers = CircuitBreakerRegistry(CircuitBreakerConfig(failure_threshold=threshold))

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    return Router(config, providers, breakers, sleep=fake_sleep, rng=lambda: 0.5)


def request() -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "chain", "messages": [{"role": "user", "content": "x"}]}
    )


async def test_first_target_success_no_fallback(sleeps: list[float]) -> None:
    a, b = ScriptedProvider("a"), ScriptedProvider("b")
    router = make_router({"a": a, "b": b, "c": ScriptedProvider("c")}, sleeps)
    routed = await router.chat(router.resolve("chain"), request())
    assert routed.result.content == "from a"
    assert routed.outcome.fallbacks == 0
    assert b.calls == 0


async def test_retries_with_exponential_backoff_then_succeeds(sleeps: list[float]) -> None:
    a = ScriptedProvider("a", [err("a"), err("a", 429, "rate_limited")])
    router = make_router({"a": a, "b": ScriptedProvider("b"), "c": ScriptedProvider("c")}, sleeps)
    routed = await router.chat(router.resolve("chain"), request())
    assert routed.result.content == "from a"
    assert a.calls == 3
    # full jitter with rng=0.5: 0.5 * min(8, 1 * 2**attempt)
    assert sleeps == [0.5, 1.0]
    assert [x.outcome for x in routed.outcome.attempts] == [
        "server_error",
        "rate_limited",
        "success",
    ]


async def test_fallback_order_after_retries_exhausted(sleeps: list[float]) -> None:
    a = ScriptedProvider("a", [err("a")] * 3)
    b = ScriptedProvider("b", [err("b", None, "timeout")] * 3)
    c = ScriptedProvider("c")
    router = make_router({"a": a, "b": b, "c": c}, sleeps)
    routed = await router.chat(router.resolve("chain"), request())
    assert routed.result.content == "from c"
    assert (a.calls, b.calls, c.calls) == (3, 3, 1)
    assert routed.outcome.fallbacks == 2
    assert [x.provider for x in routed.outcome.attempts] == ["a"] * 3 + ["b"] * 3 + ["c"]


async def test_non_retryable_failover_error_skips_retries(sleeps: list[float]) -> None:
    a = ScriptedProvider("a", [err("a", 401, "auth_error")])
    b = ScriptedProvider("b")
    router = make_router({"a": a, "b": b, "c": ScriptedProvider("c")}, sleeps)
    routed = await router.chat(router.resolve("chain"), request())
    assert routed.result.content == "from b"
    assert a.calls == 1
    assert sleeps == []


async def test_client_error_is_not_failed_over(sleeps: list[float]) -> None:
    a = ScriptedProvider("a", [err("a", 400, "client_error")])
    b = ScriptedProvider("b")
    router = make_router({"a": a, "b": b, "c": ScriptedProvider("c")}, sleeps)
    with pytest.raises(ProviderError) as info:
        await router.chat(router.resolve("chain"), request())
    assert info.value.status_code == 400
    assert b.calls == 0
    assert router.breakers.get("a").state is CircuitState.CLOSED  # 4xx says nothing about health


async def test_retry_after_longer_than_backoff_cap_fails_over_immediately(
    sleeps: list[float],
) -> None:
    a = ScriptedProvider("a", [err("a", 429, "rate_limited", retry_after=60)])
    router = make_router({"a": a, "b": ScriptedProvider("b"), "c": ScriptedProvider("c")}, sleeps)
    routed = await router.chat(router.resolve("chain"), request())
    assert routed.result.content == "from b"
    assert a.calls == 1
    assert sleeps == []


async def test_all_targets_fail(sleeps: list[float]) -> None:
    providers = {n: ScriptedProvider(n, [err(n)] * 10) for n in "abc"}
    router = make_router(providers, sleeps)
    with pytest.raises(AllTargetsFailedError) as info:
        await router.chat(router.resolve("chain"), request())
    assert len(info.value.errors) == 9
    assert info.value.to_gateway_error().status_code == 503


async def test_open_circuit_is_skipped(sleeps: list[float]) -> None:
    a = ScriptedProvider("a", [err("a")] * 100)
    b = ScriptedProvider("b")
    router = make_router({"a": a, "b": b, "c": ScriptedProvider("c")}, sleeps, threshold=3)
    await router.chat(router.resolve("chain"), request())  # 3 failures -> open
    assert router.breakers.get("a").state is CircuitState.OPEN
    calls_before = a.calls
    routed = await router.chat(router.resolve("chain"), request())
    assert a.calls == calls_before  # never called while open
    assert routed.outcome.attempts[0].outcome == "circuit_open"
    assert routed.outcome.fallbacks == 1


async def test_disabled_providers_are_skipped(sleeps: list[float]) -> None:
    config = parse_gateway_config(CONFIG, environ={})
    from llm_gateway.providers import build_providers

    providers = build_providers(config)
    c = ScriptedProvider("c")
    providers["c"] = c
    router = Router(config, providers, CircuitBreakerRegistry(config.circuit_breaker))
    routed = await router.chat(router.resolve("skip-disabled"), request())
    assert routed.result.content == "from c"
    assert routed.outcome.attempts[0].provider == "c"


async def test_stream_fails_over_before_first_chunk(sleeps: list[float]) -> None:
    a = ScriptedProvider("a", [err("a")] * 3)
    b = ScriptedProvider("b")
    router = make_router({"a": a, "b": b, "c": ScriptedProvider("c")}, sleeps)
    routed = await router.stream(router.resolve("chain"), request())
    events = [e async for e in routed]
    assert events[0].content == "from b"
    assert routed.outcome.target.provider == "b"
    assert routed.outcome.fallbacks == 1


def test_resolve_direct_and_unknown(sleeps: list[float]) -> None:
    router = make_router({n: ScriptedProvider(n) for n in "abc"}, sleeps)
    direct = router.resolve("b:any-model")
    assert direct.direct
    assert direct.config.targets[0].model == "any-model"
    with pytest.raises(GatewayError) as info:
        router.resolve("nope")
    assert info.value.status_code == 404
    with pytest.raises(GatewayError):
        router.resolve("zzz:model")
