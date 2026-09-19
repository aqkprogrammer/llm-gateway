"""Data-access functions. Callers own the session / transaction."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Integer, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from llm_gateway.db.models import ApiKey, Team, UsageRecord


async def create_team(session: AsyncSession, **fields: Any) -> Team:
    team = Team(**fields)
    session.add(team)
    await session.flush()
    return team


async def get_team(session: AsyncSession, team_id: str) -> Team | None:
    return await session.get(Team, team_id)


async def get_team_by_name(session: AsyncSession, name: str) -> Team | None:
    return await session.scalar(select(Team).where(Team.name == name))


async def list_teams(session: AsyncSession) -> Sequence[Team]:
    return (await session.scalars(select(Team).order_by(Team.created_at))).all()


async def create_api_key(session: AsyncSession, **fields: Any) -> ApiKey:
    key = ApiKey(**fields)
    session.add(key)
    await session.flush()
    return key


async def get_api_key(session: AsyncSession, key_id: str) -> ApiKey | None:
    return await session.get(ApiKey, key_id)


async def list_api_keys(session: AsyncSession, team_id: str | None = None) -> Sequence[ApiKey]:
    stmt = select(ApiKey).order_by(ApiKey.created_at)
    if team_id is not None:
        stmt = stmt.where(ApiKey.team_id == team_id)
    return (await session.scalars(stmt)).all()


async def find_api_key_by_hash(session: AsyncSession, key_hash: str) -> ApiKey | None:
    stmt = select(ApiKey).where(ApiKey.key_hash == key_hash).options(selectinload(ApiKey.team))
    return await session.scalar(stmt)


async def sum_team_spend(session: AsyncSession, team_id: str, since: datetime) -> float:
    stmt = select(func.coalesce(func.sum(UsageRecord.cost_usd), 0.0)).where(
        UsageRecord.team_id == team_id, UsageRecord.created_at >= since
    )
    return float(await session.scalar(stmt) or 0.0)


async def add_usage_records(session: AsyncSession, records: list[UsageRecord]) -> None:
    session.add_all(records)
    await session.flush()


async def usage_summary(
    session: AsyncSession, *, team_id: str | None, since: datetime
) -> list[dict[str, Any]]:
    stmt = (
        select(
            UsageRecord.team_id,
            UsageRecord.provider,
            UsageRecord.model,
            func.count().label("requests"),
            func.sum(UsageRecord.prompt_tokens).label("prompt_tokens"),
            func.sum(UsageRecord.completion_tokens).label("completion_tokens"),
            func.sum(UsageRecord.cost_usd).label("cost_usd"),
            func.sum(cast(UsageRecord.cached, Integer)).label("cache_hits"),
            func.sum(UsageRecord.fallbacks).label("fallbacks"),
        )
        .where(UsageRecord.created_at >= since)
        .group_by(UsageRecord.team_id, UsageRecord.provider, UsageRecord.model)
        .order_by(UsageRecord.team_id, UsageRecord.provider, UsageRecord.model)
    )
    if team_id is not None:
        stmt = stmt.where(UsageRecord.team_id == team_id)
    rows = (await session.execute(stmt)).mappings().all()
    return [
        {
            "team_id": row["team_id"],
            "provider": row["provider"],
            "model": row["model"],
            "requests": int(row["requests"] or 0),
            "prompt_tokens": int(row["prompt_tokens"] or 0),
            "completion_tokens": int(row["completion_tokens"] or 0),
            "cost_usd": round(float(row["cost_usd"] or 0.0), 8),
            "cache_hits": int(row["cache_hits"] or 0),
            "fallbacks": int(row["fallbacks"] or 0),
        }
        for row in rows
    ]
