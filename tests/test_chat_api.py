from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from llm_gateway.state import GatewayState
from tests.conftest import ADMIN, auth, chat_body


def parse_sse(text: str) -> list[Any]:
    out: list[Any] = []
    for block in text.strip().split("\n\n"):
        data = block.removeprefix("data: ")
        out.append(data if data == "[DONE]" else json.loads(data))
    return out


async def test_non_streaming_completion(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key = await make_key()
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "mock-large"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"].startswith("[mock-large] hello ->")
    assert (
        body["usage"]["total_tokens"]
        == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    assert r.headers["x-gateway-provider"] == "primary"
    assert r.headers["x-gateway-fallbacks"] == "0"
    assert float(r.headers["x-gateway-cost-usd"]) > 0
    assert r.headers["x-request-id"].startswith("req_")


async def test_request_id_is_propagated(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key = await make_key()
    r = await client.post(
        "/v1/chat/completions", json=chat_body(), headers={**auth(key), "x-request-id": "abc-123"}
    )
    assert r.headers["x-request-id"] == "abc-123"


async def test_streaming_completion(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key = await make_key()
    r = await client.post(
        "/v1/chat/completions",
        json=chat_body(stream=True, stream_options={"include_usage": True}),
        headers=auth(key),
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text.startswith("[mock-large] hello ->")
    finish = [
        c["choices"][0]["finish_reason"]
        for c in chunks
        if c["choices"] and c["choices"][0]["finish_reason"]
    ]
    assert finish == ["stop"]
    usage_chunk = chunks[-1]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["completion_tokens"] > 0
    assert len({c["id"] for c in chunks}) == 1


async def test_streaming_without_include_usage_omits_usage_chunk(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key = await make_key()
    r = await client.post("/v1/chat/completions", json=chat_body(stream=True), headers=auth(key))
    assert all("usage" not in c for c in parse_sse(r.text)[:-1])


async def test_failover_to_secondary(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key = await make_key()
    r = await client.post("/v1/chat/completions", json=chat_body("failover"), headers=auth(key))
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "secondary"
    assert r.headers["x-gateway-fallbacks"] == "1"
    assert r.headers["x-gateway-attempts"] == "3"  # 2 attempts on broken + 1 on secondary

    r = await client.post(
        "/v1/chat/completions", json=chat_body("failover", stream=True), headers=auth(key)
    )
    assert r.headers["x-gateway-provider"] == "secondary"
    assert parse_sse(r.text)[-1] == "[DONE]"


async def test_keyless_provider_is_skipped(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key = await make_key()
    r = await client.post("/v1/chat/completions", json=chat_body("keyless"), headers=auth(key))
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "secondary"
    assert r.headers["x-gateway-fallbacks"] == "0"


async def test_all_providers_failing_returns_503(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key = await make_key()
    await client.patch("/admin/providers/primary", json={"failure_rate": 1.0}, headers=ADMIN)
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "all_providers_failed"


async def test_unknown_model_and_validation_errors(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key = await make_key()
    r = await client.post("/v1/chat/completions", json=chat_body("gpt-17"), headers=auth(key))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
    r = await client.post(
        "/v1/chat/completions", json={"model": "demo", "messages": []}, headers=auth(key)
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "messages"
    r = await client.post("/v1/chat/completions", json=chat_body(n=2), headers=auth(key))
    assert r.status_code == 400


async def test_direct_provider_routing(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key = await make_key()
    r = await client.post(
        "/v1/chat/completions", json=chat_body("secondary:mock-small"), headers=auth(key)
    )
    assert r.status_code == 200
    assert r.json()["model"] == "mock-small"


async def test_models_endpoint(client: httpx.AsyncClient, make_key: Callable[..., Any]) -> None:
    _, key = await make_key()
    data = (await client.get("/v1/models", headers=auth(key))).json()
    ids = {m["id"]: m for m in data["data"]}
    assert {"demo", "failover", "keyless", "primary:mock-large"} <= set(ids)
    assert ids["demo"]["object"] == "model"


async def test_rate_limit_returns_429_with_headers(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key = await make_key(rpm_limit=2)
    for remaining in ("1", "0"):
        r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
        assert r.status_code == 200
        assert r.headers["x-ratelimit-limit-requests"] == "2"
        assert r.headers["x-ratelimit-remaining-requests"] == remaining
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(r.headers["retry-after"]) >= 1
    assert r.headers["x-ratelimit-remaining-requests"] == "0"


async def test_key_level_tpm_limit(client: httpx.AsyncClient, make_key: Callable[..., Any]) -> None:
    _, key = await make_key(key_fields={"tpm_limit": 20})
    first = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert first.status_code == 200
    assert "x-ratelimit-limit-tokens" in first.headers
    # Settlement charged real usage (> estimate) so the bucket is now drained.
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert r.status_code == 429
    assert "tokens" in r.json()["error"]["message"]


async def test_budget_exhaustion_returns_402(
    client: httpx.AsyncClient, make_key: Callable[..., Any], state: GatewayState
) -> None:
    team, key = await make_key(daily_budget_usd=0.0002)
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert r.status_code == 200  # ~0.0003 USD: crosses the limit
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert r.status_code == 402
    err = r.json()["error"]
    assert err["type"] == "budget_exceeded"
    assert err["code"] == "daily_budget_exceeded"
    assert "x-gateway-budget-reset" in r.headers

    await state.usage_writer.flush()
    usage = (await client.get("/admin/usage", params={"team_id": team["id"]}, headers=ADMIN)).json()
    assert usage["total_requests"] == 2  # the rejected request is logged too
    assert usage["total_cost_usd"] > 0.0002

    # Raising the budget unblocks the team.
    await client.patch(f"/admin/teams/{team['id']}", json={"daily_budget_usd": 10}, headers=ADMIN)
    assert (
        await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    ).status_code == 200


async def test_budget_survives_redis_flush(
    client: httpx.AsyncClient, make_key: Callable[..., Any], state: GatewayState
) -> None:
    _, key = await make_key(daily_budget_usd=0.0002)
    await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    await state.usage_writer.flush()
    await state.redis.flushall()
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert r.status_code == 402  # re-hydrated from the usage log


async def test_response_cache(client: httpx.AsyncClient, make_key: Callable[..., Any]) -> None:
    _, key = await make_key()
    body = chat_body(temperature=0)
    first = await client.post("/v1/chat/completions", json=body, headers=auth(key))
    second = await client.post("/v1/chat/completions", json=body, headers=auth(key))
    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "hit"
    assert second.headers["x-gateway-cost-usd"] == "0"
    assert first.json() == second.json()

    bypass = await client.post(
        "/v1/chat/completions", json=body, headers={**auth(key), "x-gateway-cache": "bypass"}
    )
    assert "x-gateway-cache" not in bypass.headers
    sampled = await client.post(
        "/v1/chat/completions", json=chat_body(temperature=0.7), headers=auth(key)
    )
    assert "x-gateway-cache" not in sampled.headers


async def test_cache_is_scoped_per_team(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    _, key_a = await make_key("team-a")
    _, key_b = await make_key("team-b")
    body = chat_body(temperature=0)
    await client.post("/v1/chat/completions", json=body, headers=auth(key_a))
    r = await client.post("/v1/chat/completions", json=body, headers=auth(key_b))
    assert r.headers["x-gateway-cache"] == "miss"


@pytest.mark.parametrize("stream", [False, True])
async def test_usage_is_persisted(
    client: httpx.AsyncClient, make_key: Callable[..., Any], state: GatewayState, stream: bool
) -> None:
    team, key = await make_key()
    await client.post(
        "/v1/chat/completions", json=chat_body("failover", stream=stream), headers=auth(key)
    )
    await state.drain()
    usage = (await client.get("/admin/usage", params={"team_id": team["id"]}, headers=ADMIN)).json()
    row = usage["rows"][0]
    assert row["provider"] == "secondary"
    assert row["model"] == "mock-small"
    assert row["requests"] == 1
    assert row["fallbacks"] == 1
    assert row["prompt_tokens"] > 0
    assert row["completion_tokens"] > 0
