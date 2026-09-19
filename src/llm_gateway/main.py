"""FastAPI application factory."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_gateway import __version__
from llm_gateway.api import admin, health, openai
from llm_gateway.config import GatewayConfig, Settings, load_gateway_config
from llm_gateway.db.session import init_db
from llm_gateway.errors import GatewayError
from llm_gateway.providers import Provider
from llm_gateway.state import GatewayState, build_state
from llm_gateway.telemetry.logging import configure_logging, request_id_var
from llm_gateway.telemetry.tracing import setup_tracing, shutdown_tracing

logger = logging.getLogger(__name__)


class RequestIdMiddleware:
    """Pure ASGI middleware (streaming-safe): propagate or mint ``x-request-id``."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming[:128] if incoming else f"req_{uuid.uuid4().hex}"
        token = request_id_var.set(request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


def _error_response(exc: GatewayError) -> JSONResponse:
    return JSONResponse(exc.to_body(), status_code=exc.status_code, headers=exc.headers)


def create_app(
    settings: Settings | None = None,
    config: GatewayConfig | None = None,
    *,
    redis: Redis | None = None,
    providers: dict[str, Provider] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_json)
    config = config or load_gateway_config(settings.config_path)
    state = build_state(settings, config, redis=redis, providers=providers)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await init_db(state.engine)
        state.usage_writer.start()
        enabled = [n for n, p in state.providers.items() if p.enabled]
        disabled = [n for n, p in state.providers.items() if not p.enabled]
        logger.info(
            "gateway started",
            extra={
                "version": __version__,
                "routes": sorted(config.routes),
                "providers_enabled": enabled,
                "providers_disabled": disabled,
                "admin_api": settings.master_key is not None,
            },
        )
        try:
            yield
        finally:
            await state.drain()
            await state.usage_writer.stop()
            for provider in state.providers.values():
                await provider.aclose()
            await state.redis.aclose()
            await state.engine.dispose()
            shutdown_tracing()

    app = FastAPI(
        title="LLM Gateway",
        version=__version__,
        description=(
            "OpenAI-compatible LLM gateway with fallback routing, per-team budgets, "
            "Redis token-bucket rate limits, and OpenTelemetry/Prometheus observability."
        ),
        lifespan=lifespan,
    )
    app.state.gateway = state

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = [str(p) for p in first.get("loc", []) if p != "body"]
        message = f"{'.'.join(loc) or 'body'}: {first.get('msg', 'invalid request')}"
        status = 400 if request.url.path.startswith("/v1/") else 422
        body = GatewayError(status, message, param=".".join(loc) or None).to_body()
        body["error"]["details"] = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in errors
        ]
        return JSONResponse(body, status_code=status)

    app.include_router(openai.router)
    app.include_router(admin.router)
    app.include_router(health.router)
    app.add_middleware(RequestIdMiddleware)

    if setup_tracing(settings) is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        # Skip per-message ASGI send/receive spans: a streamed completion would otherwise
        # produce one span per SSE chunk.
        FastAPIInstrumentor.instrument_app(
            app, excluded_urls="health/.*,metrics", exclude_spans=["send", "receive"]
        )

    return app


def gateway_state(app: FastAPI) -> GatewayState:
    return app.state.gateway
