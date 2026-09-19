from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.conftest import auth, chat_body


async def test_metrics_endpoint_exposes_gateway_series(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key = await make_key(rpm_limit=1)
    await client.post("/v1/chat/completions", json=chat_body("failover"), headers=auth(key))
    await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))  # 429

    r = await client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    for series in (
        "llm_gateway_requests_total",
        "llm_gateway_request_duration_seconds_bucket",
        "llm_gateway_provider_attempts_total",
        "llm_gateway_retries_total",
        "llm_gateway_fallbacks_total",
        "llm_gateway_tokens_total",
        "llm_gateway_cost_usd_total",
        "llm_gateway_circuit_breaker_state",
        "llm_gateway_rate_limit_rejections_total",
        "llm_gateway_in_flight_requests",
    ):
        assert series in text, series
    assert (
        'llm_gateway_fallbacks_total{from_provider="broken",reason="server_error",route="failover",to_provider="secondary"}'
        in text
    )
    assert (
        'llm_gateway_rate_limit_rejections_total{limit="requests",scope="team",team="team-a"}'
        in text
    )
    assert 'llm_gateway_circuit_breaker_state{provider="primary"} 0.0' in text


async def test_health_endpoints(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health/live")).json()["status"] == "ok"
    ready = await client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"redis": "ok", "database": "ok"}
    providers = (await client.get("/health/providers")).json()["providers"]
    assert providers["broken"]["circuit"] == "closed"


async def test_trace_spans_and_attributes(
    client: httpx.AsyncClient, make_key: Callable[..., Any], span_exporter: InMemorySpanExporter
) -> None:
    _, key = await make_key()
    await client.post("/v1/chat/completions", json=chat_body("failover"), headers=auth(key))
    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    attempts = [
        s for s in span_exporter.get_finished_spans() if s.name == "gateway.provider_attempt"
    ]
    request_span, routing_span = spans["gateway.request"], spans["gateway.routing"]

    assert routing_span.parent is not None
    assert routing_span.parent.span_id == request_span.context.span_id
    assert all(
        a.parent is not None and a.parent.span_id == routing_span.context.span_id for a in attempts
    )
    assert [a.attributes["gateway.provider"] for a in attempts] == ["broken", "broken", "secondary"]
    attrs = request_span.attributes
    assert attrs is not None
    assert attrs["gateway.team"] == "team-a"
    assert attrs["gateway.provider"] == "secondary"
    assert attrs["gateway.fallback_count"] == 1
    assert attrs["gen_ai.usage.output_tokens"] > 0
    assert attrs["gateway.cost_usd"] > 0


async def test_streaming_span_ends_after_stream(
    client: httpx.AsyncClient, make_key: Callable[..., Any], span_exporter: InMemorySpanExporter
) -> None:
    _, key = await make_key()
    await client.post("/v1/chat/completions", json=chat_body(stream=True), headers=auth(key))
    request_spans = [s for s in span_exporter.get_finished_spans() if s.name == "gateway.request"]
    assert len(request_spans) == 1
    assert request_spans[0].attributes["gen_ai.usage.output_tokens"] > 0
