from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import fakeredis
import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_gateway.config import GatewayConfig, Settings, parse_gateway_config
from llm_gateway.main import create_app
from llm_gateway.state import GatewayState

MASTER_KEY = "sk-master-test"

TEST_CONFIG = """
providers:
  primary:
    type: mock
    latency_ms: 0
    jitter_ms: 0
    stream_delay_ms: 0
    response_words: 12
    models: [mock-large]
  secondary:
    type: mock
    latency_ms: 0
    jitter_ms: 0
    stream_delay_ms: 0
    response_words: 12
  broken:
    type: mock
    latency_ms: 0
    jitter_ms: 0
    failure_rate: 1.0
    failure_status: 503
  anthropic:
    type: anthropic
    api_key: ""
routes:
  demo:
    targets: [primary:mock-large]
    max_retries: 0
  failover:
    targets: [broken:mock-large, secondary:mock-small]
    max_retries: 1
    backoff_base_s: 0
    backoff_max_s: 0
  keyless:
    targets: [anthropic:claude-sonnet-5, secondary:mock-small]
circuit_breaker:
  failure_threshold: 3
  recovery_timeout_s: 30
pricing:
  mock-large: {input: 3.0, output: 15.0}
  mock-small: {input: 0.5, output: 1.5}
"""

_EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_provider)


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    _EXPORTER.clear()
    return _EXPORTER


@pytest.fixture
def gateway_config() -> GatewayConfig:
    return parse_gateway_config(TEST_CONFIG, environ={})


@pytest.fixture
def redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'gateway.db'}",
        master_key=MASTER_KEY,
        log_json=True,
        log_level="WARNING",
        key_cache_ttl_s=0,
        usage_flush_interval_s=0.05,
        otel_enabled=False,
    )


@pytest.fixture
async def app(
    settings: Settings, gateway_config: GatewayConfig, redis: fakeredis.FakeAsyncRedis
) -> AsyncIterator[FastAPI]:
    application = create_app(settings, gateway_config, redis=redis)
    async with LifespanManager(application):
        yield application


@pytest.fixture
def state(app: FastAPI) -> GatewayState:
    return app.state.gateway


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as c:
        yield c


ADMIN = {"Authorization": f"Bearer {MASTER_KEY}"}


@pytest.fixture
def make_key(client: httpx.AsyncClient) -> Callable[..., Any]:
    async def _make(
        team: str = "team-a", *, key_fields: dict[str, Any] | None = None, **team_fields: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        r = await client.post("/admin/teams", json={"name": team, **team_fields}, headers=ADMIN)
        assert r.status_code == 201, r.text
        team_obj = r.json()
        r = await client.post(
            "/admin/keys",
            json={"team_id": team_obj["id"], "name": "k", **(key_fields or {})},
            headers=ADMIN,
        )
        assert r.status_code == 201, r.text
        return team_obj, r.json()

    return _make


def auth(key: dict[str, Any]) -> dict[str, str]:
    return {"Authorization": f"Bearer {key['key']}"}


def chat_body(model: str = "demo", content: str = "hello", **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": content}], **extra}
