"""Admin API (master key): teams, virtual keys, budgets/limits, usage, provider health."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from llm_gateway.api.deps import StateDep, require_master_key
from llm_gateway.auth import as_utc, generate_api_key, hash_api_key
from llm_gateway.db import repository
from llm_gateway.db.models import ApiKey, Team
from llm_gateway.errors import GatewayError
from llm_gateway.providers.mock import MockProvider

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_master_key)])


# -- schemas -------------------------------------------------------------------------------


class TeamLimits(BaseModel):
    daily_budget_usd: float | None = Field(default=None, ge=0)
    monthly_budget_usd: float | None = Field(default=None, ge=0)
    soft_limit_pct: float = Field(default=0.8, gt=0, le=1)
    rpm_limit: int | None = Field(default=None, ge=1)
    tpm_limit: int | None = Field(default=None, ge=1)
    allowed_models: list[str] | None = None


class TeamCreate(TeamLimits):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class TeamUpdate(BaseModel):
    """Partial update; send ``null`` to clear a limit."""

    daily_budget_usd: float | None = Field(default=None, ge=0)
    monthly_budget_usd: float | None = Field(default=None, ge=0)
    soft_limit_pct: float | None = Field(default=None, gt=0, le=1)
    rpm_limit: int | None = Field(default=None, ge=1)
    tpm_limit: int | None = Field(default=None, ge=1)
    allowed_models: list[str] | None = None
    is_active: bool | None = None


class TeamOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    daily_budget_usd: float | None
    monthly_budget_usd: float | None
    soft_limit_pct: float
    rpm_limit: int | None
    tpm_limit: int | None
    allowed_models: list[str] | None
    is_active: bool
    created_at: datetime
    spend_usd: dict[str, float] | None = None


class KeyCreate(BaseModel):
    team_id: str
    name: str = Field(default="default", min_length=1, max_length=128)
    rpm_limit: int | None = Field(default=None, ge=1)
    tpm_limit: int | None = Field(default=None, ge=1)
    allowed_models: list[str] | None = None
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)


class KeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    team_id: str
    name: str
    key_prefix: str
    rpm_limit: int | None
    tpm_limit: int | None
    allowed_models: list[str] | None
    created_at: datetime
    expires_at: datetime | None
    revoked_at: datetime | None


class KeyCreated(KeyOut):
    key: str = Field(description="The secret. Shown only once - store it now.")


class MockUpdate(BaseModel):
    failure_rate: float | None = Field(default=None, ge=0, le=1)
    failure_status: int | None = Field(default=None, ge=400, le=599)
    failure_mode: Literal["error", "timeout"] | None = None
    latency_ms: float | None = Field(default=None, ge=0)


# -- helpers -------------------------------------------------------------------------------


async def _team_or_404(session: Any, team_id: str) -> Team:
    team = await repository.get_team(session, team_id)
    if team is None:
        raise GatewayError(404, f"Team {team_id!r} not found.", code="team_not_found")
    return team


def _validate_models(state: StateDep, models: list[str] | None) -> None:
    for model in models or []:
        try:
            state.router.resolve(model)
        except GatewayError as exc:
            raise GatewayError(
                422, f"allowed_models: unknown model {model!r}.", param="allowed_models"
            ) from exc


# -- teams ---------------------------------------------------------------------------------


@router.post("/teams", status_code=status.HTTP_201_CREATED, response_model=TeamOut)
async def create_team(body: TeamCreate, state: StateDep) -> TeamOut:
    _validate_models(state, body.allowed_models)
    async with state.session_factory() as session:
        try:
            async with session.begin():
                team = await repository.create_team(session, **body.model_dump())
        except IntegrityError as exc:
            raise GatewayError(
                409, f"Team {body.name!r} already exists.", code="team_exists"
            ) from exc
    return TeamOut.model_validate(team)


@router.get("/teams", response_model=list[TeamOut])
async def list_teams(state: StateDep) -> list[TeamOut]:
    async with state.session_factory() as session:
        teams = await repository.list_teams(session)
    return [TeamOut.model_validate(t) for t in teams]


@router.get("/teams/{team_id}", response_model=TeamOut)
async def get_team(team_id: str, state: StateDep) -> TeamOut:
    async with state.session_factory() as session:
        team = await _team_or_404(session, team_id)
    out = TeamOut.model_validate(team)
    out.spend_usd = await state.budgets.spend(team.id)
    return out


@router.patch("/teams/{team_id}", response_model=TeamOut)
async def update_team(team_id: str, body: TeamUpdate, state: StateDep) -> TeamOut:
    changes = body.model_dump(exclude_unset=True)
    _validate_models(state, changes.get("allowed_models"))
    async with state.session_factory() as session, session.begin():
        team = await _team_or_404(session, team_id)
        for field_name, value in changes.items():
            if field_name in ("soft_limit_pct", "is_active") and value is None:
                continue
            setattr(team, field_name, value)
    state.key_store.invalidate(team_id=team_id)
    return TeamOut.model_validate(team)


# -- keys ----------------------------------------------------------------------------------


@router.post("/keys", status_code=status.HTTP_201_CREATED, response_model=KeyCreated)
async def create_key(body: KeyCreate, state: StateDep) -> KeyCreated:
    _validate_models(state, body.allowed_models)
    secret = generate_api_key()
    expires_at = (
        datetime.now(UTC) + timedelta(days=body.expires_in_days) if body.expires_in_days else None
    )
    async with state.session_factory() as session, session.begin():
        await _team_or_404(session, body.team_id)
        key = await repository.create_api_key(
            session,
            team_id=body.team_id,
            name=body.name,
            key_hash=hash_api_key(secret),
            key_prefix=secret[:12],
            rpm_limit=body.rpm_limit,
            tpm_limit=body.tpm_limit,
            allowed_models=body.allowed_models,
            expires_at=expires_at,
        )
    return KeyCreated(**KeyOut.model_validate(key).model_dump(), key=secret)


@router.get("/keys", response_model=list[KeyOut])
async def list_keys(
    state: StateDep, team_id: Annotated[str | None, Query()] = None
) -> list[KeyOut]:
    async with state.session_factory() as session:
        keys = await repository.list_api_keys(session, team_id)
    return [KeyOut.model_validate(k) for k in keys]


@router.delete("/keys/{key_id}", response_model=KeyOut)
async def revoke_key(key_id: str, state: StateDep) -> KeyOut:
    async with state.session_factory() as session, session.begin():
        key: ApiKey | None = await repository.get_api_key(session, key_id)
        if key is None:
            raise GatewayError(404, f"Key {key_id!r} not found.", code="key_not_found")
        if key.revoked_at is None:
            key.revoked_at = datetime.now(UTC)
    state.key_store.invalidate(key_id=key_id)
    out = KeyOut.model_validate(key)
    out.revoked_at = as_utc(out.revoked_at)
    return out


# -- usage ---------------------------------------------------------------------------------


@router.get("/usage")
async def usage(
    state: StateDep,
    team_id: Annotated[str | None, Query()] = None,
    days: Annotated[int, Query(ge=1, le=366)] = 30,
) -> dict[str, Any]:
    await state.usage_writer.flush()
    since = datetime.now(UTC) - timedelta(days=days)
    async with state.session_factory() as session:
        rows = await repository.usage_summary(session, team_id=team_id, since=since)
    return {
        "since": since.isoformat(),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 8),
        "total_requests": sum(r["requests"] for r in rows),
        "rows": rows,
    }


# -- providers -----------------------------------------------------------------------------


@router.get("/providers")
async def providers(state: StateDep) -> dict[str, Any]:
    breakers = state.breakers.snapshot()
    return {
        name: {
            "type": provider.type,
            "enabled": provider.enabled,
            "base_url": provider.base_url or None,
            "circuit": breakers.get(name),
        }
        for name, provider in state.providers.items()
    }


@router.post("/providers/{name}/reset")
async def reset_provider(name: str, state: StateDep) -> dict[str, Any]:
    if name not in state.providers:
        raise GatewayError(404, f"Provider {name!r} not found.", code="provider_not_found")
    breaker = state.breakers.get(name)
    breaker.reset()
    return {"provider": name, "circuit": breaker.snapshot()}


@router.patch("/providers/{name}")
async def update_mock_provider(name: str, body: MockUpdate, state: StateDep) -> dict[str, Any]:
    """Chaos controls for ``mock`` providers (failure injection / latency) at runtime."""
    provider = state.providers.get(name)
    if provider is None:
        raise GatewayError(404, f"Provider {name!r} not found.", code="provider_not_found")
    if not isinstance(provider, MockProvider):
        raise GatewayError(409, "Only mock providers can be reconfigured at runtime.")
    provider.update(**body.model_dump(exclude_none=True))
    cfg = provider.config
    return {
        "provider": name,
        "failure_rate": cfg.failure_rate,
        "failure_status": cfg.failure_status,
        "failure_mode": cfg.failure_mode,
        "latency_ms": cfg.latency_ms,
    }
