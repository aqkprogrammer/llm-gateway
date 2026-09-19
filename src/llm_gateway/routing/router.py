"""The routing engine: ordered fallback chains with retries, backoff and circuit breakers.

For each target in a route, in order:

1. Skip it if the provider is disabled (e.g. no API key) or its circuit is open.
2. Call it with the route timeout. On a *retryable* error (429, 5xx, timeout, connection)
   retry the same target with exponential backoff and full jitter, honouring upstream
   ``Retry-After`` when it fits inside the backoff cap.
3. On a *failover* error (retries exhausted, or 401/403/404 which retrying won't fix)
   move on to the next target.
4. On a *client* error (400/413/422) stop immediately - another provider would reject
   the same payload - and surface the upstream message.

Streams can only fail over before the first chunk reaches the client. The router
therefore awaits the first event of each attempt inside the timeout; once a stream has
produced its first event it is committed to that provider.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from opentelemetry.trace import Status, StatusCode

from llm_gateway.config import GatewayConfig, RouteConfig, RouteTarget
from llm_gateway.errors import AllTargetsFailedError, GatewayError, ProviderError
from llm_gateway.providers.base import Provider
from llm_gateway.routing.circuit_breaker import CircuitBreakerRegistry
from llm_gateway.schemas import ChatCompletionRequest, ChatResult, StreamEvent
from llm_gateway.telemetry import metrics
from llm_gateway.telemetry.tracing import tracer

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(slots=True)
class ResolvedRoute:
    name: str
    config: RouteConfig
    direct: bool = False


@dataclass(slots=True)
class AttemptRecord:
    provider: str
    model: str
    attempt: int
    outcome: str
    duration_s: float
    error: str | None = None


@dataclass(slots=True)
class RoutingOutcome:
    target: RouteTarget
    provider_type: str
    attempts: list[AttemptRecord] = field(default_factory=list)
    fallbacks: int = 0


@dataclass(slots=True)
class RoutedResult:
    result: ChatResult
    outcome: RoutingOutcome


class RoutedStream:
    """A committed upstream stream: the already-received first event plus the rest."""

    def __init__(
        self,
        first: StreamEvent,
        rest: AsyncIterator[StreamEvent],
        outcome: RoutingOutcome,
        on_error: Callable[[ProviderError], None],
    ) -> None:
        self.first = first
        self._rest = rest
        self.outcome = outcome
        self._on_error = on_error

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        yield self.first
        try:
            async for event in self._rest:
                yield event
        except ProviderError as exc:
            self._on_error(exc)
            raise

    async def aclose(self) -> None:
        aclose = getattr(self._rest, "aclose", None)
        if aclose is not None:
            await aclose()


class Router:
    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider],
        breakers: CircuitBreakerRegistry,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self.providers = providers
        self.breakers = breakers
        self._sleep = sleep
        self._rng = rng

    # -- resolution ------------------------------------------------------------------------

    def resolve(self, model: str) -> ResolvedRoute:
        route = self.config.routes.get(model)
        if route is not None:
            return ResolvedRoute(name=model, config=route)
        if self.config.allow_direct_routing and ":" in model:
            target = RouteTarget.parse(model)
            if target.provider in self.providers:
                return ResolvedRoute(
                    name=model, config=self.config.direct_route(target), direct=True
                )
        raise GatewayError(
            404,
            f"The model {model!r} does not exist or is not routable by this gateway.",
            code="model_not_found",
            param="model",
        )

    def backoff_delay(
        self, route: RouteConfig, attempt: int, retry_after: float | None
    ) -> float | None:
        """Full-jitter exponential backoff. ``None`` means "don't wait - fail over"."""
        if retry_after is not None:
            return retry_after if retry_after <= route.backoff_max_s else None
        ceiling = min(route.backoff_max_s, route.backoff_base_s * (2**attempt))
        return self._rng() * ceiling

    # -- execution -------------------------------------------------------------------------

    async def chat(self, route: ResolvedRoute, request: ChatCompletionRequest) -> RoutedResult:
        async def invoke(provider: Provider, model: str, timeout: float) -> ChatResult:
            return await provider.chat(request, model, timeout=timeout)

        result, outcome = await self._run(route, invoke, stream=False)
        return RoutedResult(result=result, outcome=outcome)

    async def stream(self, route: ResolvedRoute, request: ChatCompletionRequest) -> RoutedStream:
        async def invoke(
            provider: Provider, model: str, timeout: float
        ) -> tuple[StreamEvent, AsyncIterator[StreamEvent]]:
            iterator = provider.stream(request, model, timeout=timeout)
            try:
                first = await anext(iterator)
            except StopAsyncIteration:
                raise ProviderError(
                    "upstream stream ended without data",
                    provider=provider.name,
                    status_code=502,
                    kind="server_error",
                ) from None
            except BaseException:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
                raise
            return first, iterator

        (first, rest), outcome = await self._run(route, invoke, stream=True)
        breaker = self.breakers.get(outcome.target.provider)
        return RoutedStream(first, rest, outcome, lambda exc: breaker.record_failure(exc.message))

    async def _run(
        self,
        route: ResolvedRoute,
        invoke: Callable[[Provider, str, float], Awaitable[T]],
        *,
        stream: bool,
    ) -> tuple[T, RoutingOutcome]:
        cfg = route.config
        errors: list[ProviderError] = []
        attempts: list[AttemptRecord] = []
        fallbacks = 0
        previous_failure: tuple[str, str] | None = None  # (provider, reason)

        with tracer.start_as_current_span(
            "gateway.routing",
            attributes={
                "gateway.route": route.name,
                "gateway.route.targets": [str(t) for t in cfg.targets],
                "gateway.stream": stream,
            },
        ) as routing_span:
            for target in cfg.targets:
                provider = self.providers.get(target.provider)
                if provider is None or not provider.enabled:
                    # Configuration state, not a runtime failure: skipped silently.
                    continue
                breaker = self.breakers.get(target.provider)
                attempted = False

                for attempt in range(cfg.max_retries + 1):
                    if not breaker.allow_request():
                        error = ProviderError(
                            "circuit breaker open", provider=target.provider, kind="circuit_open"
                        )
                        metrics.PROVIDER_ATTEMPTS.labels(
                            target.provider, target.model, "circuit_open"
                        ).inc()
                        attempts.append(
                            AttemptRecord(
                                target.provider, target.model, attempt, "circuit_open", 0.0
                            )
                        )
                        errors.append(error)
                        previous_failure = (target.provider, "circuit_open")
                        break

                    if not attempted and previous_failure is not None:
                        fallbacks += 1
                        metrics.FALLBACKS.labels(
                            route.name, previous_failure[0], target.provider, previous_failure[1]
                        ).inc()
                    attempted = True
                    if attempt > 0:
                        metrics.RETRIES.labels(target.provider, target.model).inc()

                    started = time.perf_counter()
                    with tracer.start_as_current_span(
                        "gateway.provider_attempt",
                        attributes={
                            "gateway.provider": target.provider,
                            "gen_ai.system": provider.type,
                            "gen_ai.request.model": target.model,
                            "gateway.attempt": attempt,
                            "gateway.fallback_index": fallbacks,
                        },
                    ) as span:
                        try:
                            value = await asyncio.wait_for(
                                invoke(provider, target.model, cfg.timeout_s), cfg.timeout_s
                            )
                        except ProviderError as exc:
                            error = exc
                        except TimeoutError:
                            error = ProviderError(
                                f"timed out after {cfg.timeout_s:g}s",
                                provider=target.provider,
                                kind="timeout",
                            )
                        except asyncio.CancelledError:
                            breaker.release()
                            raise
                        except Exception as exc:
                            logger.exception("provider raised unexpected error")
                            error = ProviderError(
                                f"unexpected provider error: {exc!r}",
                                provider=target.provider,
                                kind="internal",
                            )
                        else:
                            elapsed = time.perf_counter() - started
                            breaker.record_success()
                            metrics.PROVIDER_ATTEMPTS.labels(
                                target.provider, target.model, "success"
                            ).inc()
                            metrics.PROVIDER_LATENCY.labels(
                                target.provider, target.model, "success"
                            ).observe(elapsed)
                            if stream:
                                metrics.TIME_TO_FIRST_TOKEN.labels(
                                    target.provider, target.model
                                ).observe(elapsed)
                            span.set_attribute("gateway.outcome", "success")
                            attempts.append(
                                AttemptRecord(
                                    target.provider, target.model, attempt, "success", elapsed
                                )
                            )
                            routing_span.set_attributes(
                                {
                                    "gateway.provider": target.provider,
                                    "gen_ai.request.model": target.model,
                                    "gateway.fallback_count": fallbacks,
                                    "gateway.attempt_count": len(attempts),
                                }
                            )
                            outcome = RoutingOutcome(
                                target=target,
                                provider_type=provider.type,
                                attempts=attempts,
                                fallbacks=fallbacks,
                            )
                            return value, outcome

                        elapsed = time.perf_counter() - started
                        if error.counts_against_breaker:
                            breaker.record_failure(error.message)
                        else:
                            breaker.release()
                        metrics.PROVIDER_ATTEMPTS.labels(
                            target.provider, target.model, error.kind
                        ).inc()
                        metrics.PROVIDER_LATENCY.labels(
                            target.provider, target.model, error.kind
                        ).observe(elapsed)
                        span.set_attributes(
                            {
                                "gateway.outcome": error.kind,
                                "http.response.status_code": error.status_code or 0,
                            }
                        )
                        span.set_status(Status(StatusCode.ERROR, error.message))
                        attempts.append(
                            AttemptRecord(
                                target.provider,
                                target.model,
                                attempt,
                                error.kind,
                                elapsed,
                                error.message,
                            )
                        )
                    logger.warning(
                        "provider attempt failed",
                        extra={
                            "route": route.name,
                            "provider": target.provider,
                            "model": target.model,
                            "attempt": attempt,
                            "error_kind": error.kind,
                            "status_code": error.status_code,
                            "error": error.message,
                        },
                    )

                    if not error.failover:
                        routing_span.set_status(Status(StatusCode.ERROR, error.message))
                        raise error
                    errors.append(error)
                    previous_failure = (target.provider, error.kind)
                    if not error.retryable or attempt >= cfg.max_retries:
                        break
                    delay = self.backoff_delay(cfg, attempt, error.retry_after)
                    if delay is None:
                        break
                    await self._sleep(delay)

            routing_span.set_attribute("gateway.fallback_count", fallbacks)
            routing_span.set_status(Status(StatusCode.ERROR, "all targets failed"))
            raise AllTargetsFailedError(route.name, errors)
