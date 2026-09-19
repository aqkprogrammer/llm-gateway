"""Chat completion orchestration: the request lifecycle end to end.

authenticate -> resolve route -> budget check -> rate limit -> cache lookup
  -> route (retries / failover / circuit breakers) -> respond
  -> account (cost, spend, TPM settlement, metrics, usage log, trace attributes)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from starlette.responses import Response

from llm_gateway.auth import AuthContext
from llm_gateway.budget import BudgetExceededError, period_reset
from llm_gateway.cache import cache_key, is_cacheable
from llm_gateway.db.models import UsageRecord
from llm_gateway.errors import AllTargetsFailedError, GatewayError, ProviderError
from llm_gateway.ratelimit import Bucket
from llm_gateway.routing.router import ResolvedRoute, RoutedStream, RoutingOutcome
from llm_gateway.schemas import (
    ChatCompletionRequest,
    Usage,
    estimate_prompt_tokens,
    new_completion_id,
)
from llm_gateway.state import GatewayState
from llm_gateway.telemetry import metrics
from llm_gateway.telemetry.tracing import tracer

logger = logging.getLogger("llm_gateway.access")


@dataclass(slots=True)
class _Accounting:
    auth: AuthContext
    route: str
    request_id: str
    stream: bool
    started: float = field(default_factory=time.perf_counter)
    estimate: int = 0
    buckets: list[Bucket] = field(default_factory=list)
    tokens_acquired: bool = False
    outcome: RoutingOutcome | None = None
    accounted: bool = False


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


class ChatService:
    def __init__(self, state: GatewayState) -> None:
        self.state = state

    async def complete(
        self,
        body: ChatCompletionRequest,
        auth: AuthContext,
        *,
        request_id: str,
        bypass_cache: bool = False,
    ) -> Response:
        if body.n is not None and body.n > 1:
            raise GatewayError(400, "Only n=1 is supported by this gateway.", param="n")
        try:
            route = self.state.router.resolve(body.model)
        except GatewayError:
            metrics.REQUESTS.labels(
                "_unknown", "none", "none", "404", str(body.stream).lower()
            ).inc()
            raise
        if not auth.can_use(route.name):
            raise GatewayError(
                403,
                f"This API key is not allowed to use model {route.name!r}.",
                type="permission_error",
                code="model_not_allowed",
                param="model",
            )

        acct = _Accounting(auth=auth, route=route.name, request_id=request_id, stream=body.stream)
        span = tracer.start_span(
            "gateway.request",
            attributes={
                "gateway.request_id": request_id,
                "gateway.team": auth.team_name,
                "gateway.team_id": auth.team_id,
                "gateway.key_id": auth.key_id,
                "gateway.route": route.name,
                "gateway.stream": body.stream,
                "gen_ai.operation.name": "chat",
            },
        )
        metrics.IN_FLIGHT.inc()
        try:
            with trace.use_span(span, end_on_exit=False, record_exception=False):
                response = await self._handle(body, route, acct, span, bypass_cache)
        except GatewayError as exc:
            await self._account(
                acct, span, status=exc.status_code, usage=Usage(), error=exc.message
            )
            span.end()
            raise
        except asyncio.CancelledError:
            self.state.spawn(self._account_detached(acct, span, 499, "client disconnected"))
            raise
        except Exception as exc:
            await self._account(acct, span, status=500, usage=Usage(), error=repr(exc))
            span.end()
            raise
        if not isinstance(response, StreamingResponse):
            span.end()
        return response

    # -- request lifecycle -----------------------------------------------------------------

    async def _handle(
        self,
        body: ChatCompletionRequest,
        route: ResolvedRoute,
        acct: _Accounting,
        span: Span,
        bypass_cache: bool,
    ) -> Response:
        state = self.state
        auth = acct.auth

        # 1. Budget (cheap, read-only) before we spend a rate-limit slot.
        try:
            await state.budgets.check(auth.budget)
        except BudgetExceededError as exc:
            reset = period_reset(exc.period, datetime.now(UTC))
            raise GatewayError(
                402,
                f"Team {auth.team_name!r} has exhausted its {exc.period} budget "
                f"(${exc.spent:.4f} of ${exc.limit:.4f}). Resets at {reset.isoformat()}.",
                type="budget_exceeded",
                code=f"{exc.period}_budget_exceeded",
                headers={"x-gateway-budget-reset": reset.isoformat()},
            ) from exc

        # 2. Rate limits: RPM + TPM, per key and per team, atomically.
        acct.estimate = estimate_prompt_tokens(body)
        acct.buckets = state.rate_limiter.buckets(
            key_id=auth.key_id,
            team_id=auth.team_id,
            key_rpm=auth.key_rpm,
            key_tpm=auth.key_tpm,
            team_rpm=auth.team_rpm,
            team_tpm=auth.team_tpm,
        )
        decision = await state.rate_limiter.acquire(acct.buckets, tokens=acct.estimate)
        if not decision.allowed:
            limiting = decision.limiting
            scope = limiting.scope if limiting else "unknown"
            kind = limiting.kind if limiting else "unknown"
            metrics.RATE_LIMIT_REJECTIONS.labels(auth.team_name, scope, kind).inc()
            raise GatewayError(
                429,
                f"Rate limit exceeded: {scope} {kind} per minute "
                f"(limit {limiting.limit if limiting else '?'}). "
                f"Retry after {decision.retry_after_s:.2f}s.",
                type="rate_limit_error",
                code="rate_limit_exceeded",
                headers=decision.headers,
            )
        acct.tokens_acquired = True
        headers: dict[str, str] = {**decision.headers, "x-gateway-route": route.name}

        # 3. Exact-match cache (deterministic, non-streaming requests only).
        key: str | None = None
        if state.cache.enabled and not bypass_cache and is_cacheable(body):
            key = cache_key(auth.team_id, route.name, body)
            cached = await state.cache.get(key)
            span.set_attribute("gateway.cache_hit", cached is not None)
            if cached is not None:
                await self._account(acct, span, status=200, usage=Usage(), cached=True)
                headers.update(
                    {
                        "x-gateway-cache": "hit",
                        "x-gateway-provider": "cache",
                        "x-gateway-cost-usd": "0",
                    }
                )
                return JSONResponse(cached, headers=headers)
            headers["x-gateway-cache"] = "miss"

        # 4. Route.
        if body.stream:
            return await self._stream(body, route, acct, span, headers)
        try:
            routed = await state.router.chat(route, body)
        except ProviderError as exc:
            raise _upstream_rejection(exc) from exc
        except AllTargetsFailedError as exc:
            raise exc.to_gateway_error() from exc

        acct.outcome = routed.outcome
        result = routed.result
        payload = result.to_openai()
        cost = await self._account(acct, span, status=200, usage=result.usage)
        if key is not None:
            await state.cache.set(key, payload)
        headers.update(_routing_headers(routed.outcome))
        headers["x-gateway-cost-usd"] = f"{cost:.8f}"
        return JSONResponse(payload, headers=headers)

    async def _stream(
        self,
        body: ChatCompletionRequest,
        route: ResolvedRoute,
        acct: _Accounting,
        span: Span,
        headers: dict[str, str],
    ) -> StreamingResponse:
        try:
            routed = await self.state.router.stream(route, body)
        except ProviderError as exc:
            raise _upstream_rejection(exc) from exc
        except AllTargetsFailedError as exc:
            raise exc.to_gateway_error() from exc
        acct.outcome = routed.outcome
        headers.update(_routing_headers(routed.outcome))
        headers.update({"cache-control": "no-cache", "x-accel-buffering": "no"})
        return StreamingResponse(
            self._stream_body(routed, acct, span, include_usage=body.include_usage),
            media_type="text/event-stream",
            headers=headers,
        )

    async def _stream_body(
        self, routed: RoutedStream, acct: _Accounting, span: Span, *, include_usage: bool
    ) -> AsyncIterator[str]:
        completion_id = new_completion_id()
        created = int(time.time())
        model = routed.outcome.target.model
        usage: Usage | None = None
        chars = 0
        status = 200
        error: str | None = None
        clean_exit = False

        def chunk(delta: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
                ],
            }

        try:
            yield _sse(chunk({"role": "assistant", "content": ""}, None))
            async for event in routed:
                if event.model:
                    model = event.model
                if event.usage is not None:
                    usage = event.usage
                if not event.has_delta:
                    continue
                delta: dict[str, Any] = {}
                if event.content is not None:
                    delta["content"] = event.content
                    chars += len(event.content)
                if event.tool_calls:
                    delta["tool_calls"] = event.tool_calls
                yield _sse(chunk(delta, event.finish_reason))
            if include_usage:
                final = chunk({}, None)
                final["choices"] = []
                final["usage"] = (usage or Usage()).to_dict()
                yield _sse(final)
            yield "data: [DONE]\n\n"
            clean_exit = True
        except ProviderError as exc:
            # Headers are already sent; report the failure in-band. OpenAI SDKs raise an
            # APIError when a chunk carries an "error" object.
            status, error = 502, exc.message
            yield _sse(
                {
                    "error": {
                        "message": f"upstream stream failed: {exc.message}",
                        "type": "api_error",
                        "param": None,
                        "code": "upstream_stream_error",
                    }
                }
            )
            clean_exit = True
        finally:
            if usage is None:
                # Upstream never reported usage (disconnect / failure): estimate it.
                usage = Usage(prompt_tokens=acct.estimate, completion_tokens=max(0, chars // 4))
            if not clean_exit:
                status, error = 499, error or "client disconnected"
            span.set_attribute("gen_ai.response.model", model)
            if clean_exit:
                await routed.aclose()
                await self._account(acct, span, status=status, usage=usage, error=error)
                span.end()
            else:
                # Cancelled (client went away): finish bookkeeping outside the cancel scope.
                self.state.spawn(self._finish_detached(routed, acct, span, status, usage, error))

    async def _account_detached(
        self, acct: _Accounting, span: Span, status: int, error: str
    ) -> None:
        await self._account(acct, span, status=status, usage=Usage(), error=error)
        span.end()

    async def _finish_detached(
        self,
        routed: RoutedStream,
        acct: _Accounting,
        span: Span,
        status: int,
        usage: Usage,
        error: str | None,
    ) -> None:
        try:
            await routed.aclose()
        except Exception:
            logger.debug("error closing upstream stream", exc_info=True)
        await self._account(acct, span, status=status, usage=usage, error=error)
        span.end()

    # -- accounting ------------------------------------------------------------------------

    async def _account(
        self,
        acct: _Accounting,
        span: Span,
        *,
        status: int,
        usage: Usage,
        cached: bool = False,
        error: str | None = None,
    ) -> float:
        """Record cost, spend, TPM settlement, metrics, usage log, and trace attributes."""
        if acct.accounted:
            return 0.0
        acct.accounted = True
        metrics.IN_FLIGHT.dec()
        state = self.state
        auth = acct.auth
        outcome = acct.outcome
        provider = "cache" if cached else (outcome.target.provider if outcome else "none")
        model = outcome.target.model if outcome else "none"
        latency = time.perf_counter() - acct.started

        cost = 0.0
        if outcome is not None and not cached:
            cost = state.pricing.cost(outcome.target.provider, outcome.target.model, usage)
        try:
            if cost > 0:
                await state.budgets.record(auth.budget, cost)
            if acct.tokens_acquired:
                await state.rate_limiter.settle(
                    acct.buckets, delta_tokens=usage.total_tokens - acct.estimate
                )
        except Exception:
            logger.exception("post-request accounting failed")

        status_label = str(status)
        metrics.REQUESTS.labels(
            acct.route, provider, model, status_label, str(acct.stream).lower()
        ).inc()
        metrics.REQUEST_LATENCY.labels(acct.route, provider, model, status_label).observe(latency)
        if usage.total_tokens and outcome is not None:
            metrics.TOKENS.labels(auth.team_name, provider, model, "prompt").inc(
                usage.prompt_tokens
            )
            metrics.TOKENS.labels(auth.team_name, provider, model, "completion").inc(
                usage.completion_tokens
            )
        if cost > 0:
            metrics.COST.labels(auth.team_name, provider, model).inc(cost)

        fallbacks = outcome.fallbacks if outcome else 0
        state.usage_writer.submit(
            UsageRecord(
                request_id=acct.request_id,
                team_id=auth.team_id,
                key_id=auth.key_id,
                route=acct.route,
                provider=None if provider == "none" else provider,
                model=None if model == "none" else model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=cost,
                latency_ms=round(latency * 1000, 2),
                status_code=status,
                streamed=acct.stream,
                cached=cached,
                fallbacks=fallbacks,
            )
        )

        span.set_attributes(
            {
                "gateway.provider": provider,
                "gen_ai.request.model": model,
                "gen_ai.usage.input_tokens": usage.prompt_tokens,
                "gen_ai.usage.output_tokens": usage.completion_tokens,
                "gateway.cost_usd": cost,
                "gateway.fallback_count": fallbacks,
                "gateway.cache_hit": cached,
                "http.response.status_code": status,
            }
        )
        if status >= 400:
            span.set_status(Status(StatusCode.ERROR, error or f"HTTP {status}"))

        log = logger.warning if status >= 500 else logger.info
        log(
            "chat completion",
            extra={
                "team": auth.team_name,
                "key_id": auth.key_id,
                "route": acct.route,
                "provider": provider,
                "model": model,
                "status": status,
                "stream": acct.stream,
                "latency_ms": round(latency * 1000, 1),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost_usd": round(cost, 8),
                "fallbacks": fallbacks,
                "attempts": len(outcome.attempts) if outcome else 0,
                "cached": cached,
                **({"error": error} if error else {}),
            },
        )
        return cost


def _routing_headers(outcome: RoutingOutcome) -> dict[str, str]:
    return {
        "x-gateway-provider": outcome.target.provider,
        "x-gateway-model": outcome.target.model,
        "x-gateway-fallbacks": str(outcome.fallbacks),
        "x-gateway-attempts": str(len(outcome.attempts)),
    }


def _upstream_rejection(exc: ProviderError) -> GatewayError:
    status = exc.status_code if exc.status_code and 400 <= exc.status_code < 500 else 400
    return GatewayError(
        status,
        f"Upstream provider {exc.provider!r} rejected the request: {exc.message}",
        type="invalid_request_error",
        code="upstream_invalid_request",
    )
