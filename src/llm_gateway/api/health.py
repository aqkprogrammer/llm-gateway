"""Liveness, readiness, provider health, and Prometheus exposition."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text
from starlette.responses import Response

from llm_gateway import __version__
from llm_gateway.api.deps import StateDep

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Liveness probe")
async def live() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/health/ready", summary="Readiness probe (database + redis)")
async def ready(state: StateDep) -> Response:
    checks: dict[str, str] = {}
    try:
        await state.redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {type(exc).__name__}"
    try:
        async with state.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"
    healthy = all(v == "ok" for v in checks.values())
    return JSONResponse(
        {"status": "ok" if healthy else "degraded", "checks": checks},
        status_code=200 if healthy else 503,
    )


@router.get("/health/providers", summary="Provider availability and circuit-breaker state")
async def providers(state: StateDep) -> dict[str, Any]:
    breakers = state.breakers.snapshot()
    return {
        "providers": {
            name: {
                "type": provider.type,
                "enabled": provider.enabled,
                "circuit": breakers.get(name, {}).get("state"),
            }
            for name, provider in state.providers.items()
        }
    }


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
