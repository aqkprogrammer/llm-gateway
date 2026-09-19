from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from llm_gateway.state import GatewayState
from tests.conftest import ADMIN, auth, chat_body


async def test_admin_requires_master_key(client: httpx.AsyncClient) -> None:
    assert (await client.get("/admin/teams")).status_code == 401
    r = await client.get("/admin/teams", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_master_key"
    assert (await client.get("/admin/teams", headers=ADMIN)).status_code == 200


async def test_team_crud_and_validation(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/admin/teams",
        json={
            "name": "research",
            "monthly_budget_usd": 50,
            "rpm_limit": 100,
            "allowed_models": ["demo"],
        },
        headers=ADMIN,
    )
    assert r.status_code == 201
    team = r.json()
    assert team["id"].startswith("team_")
    assert team["soft_limit_pct"] == 0.8

    assert (
        await client.post("/admin/teams", json={"name": "research"}, headers=ADMIN)
    ).status_code == 409
    bad = await client.post(
        "/admin/teams", json={"name": "x", "allowed_models": ["nope"]}, headers=ADMIN
    )
    assert bad.status_code == 422

    r = await client.patch(
        f"/admin/teams/{team['id']}", json={"daily_budget_usd": 5, "rpm_limit": None}, headers=ADMIN
    )
    assert r.status_code == 200
    assert r.json()["daily_budget_usd"] == 5
    assert r.json()["rpm_limit"] is None

    detail = (await client.get(f"/admin/teams/{team['id']}", headers=ADMIN)).json()
    assert detail["spend_usd"] == {"daily": 0.0, "monthly": 0.0}
    assert (await client.get("/admin/teams/team_missing", headers=ADMIN)).status_code == 404


async def test_key_lifecycle_hashing_and_revocation(
    client: httpx.AsyncClient, make_key: Callable[..., Any], state: GatewayState
) -> None:
    team, key = await make_key()
    assert key["key"].startswith("sk-gw-")
    assert key["key_prefix"] == key["key"][:12]

    listed = (await client.get("/admin/keys", params={"team_id": team["id"]}, headers=ADMIN)).json()
    assert len(listed) == 1
    assert "key" not in listed[0]

    # Only the hash is stored.
    from llm_gateway.auth import hash_api_key
    from llm_gateway.db.models import ApiKey

    async with state.session_factory() as session:
        stored = await session.get(ApiKey, key["id"])
    assert stored is not None
    assert stored.key_hash == hash_api_key(key["key"])
    assert key["key"] not in stored.key_hash

    assert (
        await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    ).status_code == 200

    revoked = await client.delete(f"/admin/keys/{key['id']}", headers=ADMIN)
    assert revoked.status_code == 200
    assert revoked.json()["revoked_at"]
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert r.status_code == 401
    assert "revoked" in r.json()["error"]["message"]


async def test_invalid_and_missing_keys(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/chat/completions", json=chat_body())
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    r = await client.post(
        "/v1/chat/completions", json=chat_body(), headers={"Authorization": "Bearer sk-gw-nope"}
    )
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"


async def test_disabled_team_and_model_allowlist(
    client: httpx.AsyncClient, make_key: Callable[..., Any]
) -> None:
    team, key = await make_key(allowed_models=["demo"])
    r = await client.post("/v1/chat/completions", json=chat_body("failover"), headers=auth(key))
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "model_not_allowed"
    models = (await client.get("/v1/models", headers=auth(key))).json()["data"]
    assert [m["id"] for m in models] == ["demo"]

    await client.patch(f"/admin/teams/{team['id']}", json={"is_active": False}, headers=ADMIN)
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert r.status_code == 403


async def test_expired_key(
    client: httpx.AsyncClient, make_key: Callable[..., Any], state: GatewayState
) -> None:
    from datetime import UTC, datetime, timedelta

    from llm_gateway.db.models import ApiKey

    _, key = await make_key()
    async with state.session_factory() as session, session.begin():
        row = await session.get(ApiKey, key["id"])
        assert row is not None
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert r.status_code == 401
    assert "expired" in r.json()["error"]["message"]


async def test_provider_admin_endpoints(client: httpx.AsyncClient) -> None:
    providers = (await client.get("/admin/providers", headers=ADMIN)).json()
    assert providers["anthropic"]["enabled"] is False
    assert providers["primary"]["circuit"]["state"] == "closed"

    r = await client.patch("/admin/providers/primary", json={"failure_rate": 1.0}, headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["failure_rate"] == 1.0
    assert (
        await client.patch("/admin/providers/anthropic", json={"failure_rate": 1}, headers=ADMIN)
    ).status_code == 409
    assert (await client.post("/admin/providers/primary/reset", headers=ADMIN)).status_code == 200
    assert (await client.post("/admin/providers/nope/reset", headers=ADMIN)).status_code == 404
