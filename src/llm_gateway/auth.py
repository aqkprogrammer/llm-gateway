"""Virtual API keys: generation, hashing, lookup (with a short in-process cache)."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_gateway.budget import TeamBudget
from llm_gateway.db import repository
from llm_gateway.db.models import ApiKey
from llm_gateway.errors import GatewayError
from llm_gateway.telemetry import metrics

KEY_PREFIX = "sk-gw-"


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    # Keys are 256-bit random tokens, so a fast unsalted hash is sufficient: there is no
    # dictionary to attack, and lookups must be O(1) on the hot path.
    return hashlib.sha256(key.encode()).hexdigest()


def as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class AuthContext:
    """Immutable snapshot of the calling key and its team."""

    key_id: str
    key_name: str
    team_id: str
    team_name: str
    key_rpm: int | None
    key_tpm: int | None
    team_rpm: int | None
    team_tpm: int | None
    key_allowed_models: tuple[str, ...] | None
    team_allowed_models: tuple[str, ...] | None
    budget: TeamBudget
    expires_at: datetime | None

    def can_use(self, route: str) -> bool:
        for allowed in (self.key_allowed_models, self.team_allowed_models):
            if allowed is not None and route not in allowed:
                return False
        return True

    @classmethod
    def from_orm(cls, key: ApiKey) -> AuthContext:
        team = key.team
        return cls(
            key_id=key.id,
            key_name=key.name,
            team_id=team.id,
            team_name=team.name,
            key_rpm=key.rpm_limit,
            key_tpm=key.tpm_limit,
            team_rpm=team.rpm_limit,
            team_tpm=team.tpm_limit,
            key_allowed_models=tuple(key.allowed_models) if key.allowed_models else None,
            team_allowed_models=tuple(team.allowed_models) if team.allowed_models else None,
            budget=TeamBudget(
                team_id=team.id,
                team_name=team.name,
                daily_usd=team.daily_budget_usd,
                monthly_usd=team.monthly_budget_usd,
                soft_limit_pct=team.soft_limit_pct,
            ),
            expires_at=as_utc(key.expires_at),
        )


def _unauthorized(message: str, reason: str) -> GatewayError:
    metrics.AUTH_FAILURES.labels(reason).inc()
    return GatewayError(
        401,
        message,
        type="authentication_error",
        code="invalid_api_key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def bearer_token(authorization: str | None, x_api_key: str | None = None) -> str | None:
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return x_api_key.strip() if x_api_key else None


class KeyStore:
    """Resolves bearer tokens to :class:`AuthContext`, caching positive lookups briefly."""

    def __init__(self, session_factory: async_sessionmaker, *, ttl_s: float = 10.0) -> None:
        self._session_factory = session_factory
        self._ttl_s = ttl_s
        self._cache: dict[str, tuple[float, AuthContext]] = {}

    def invalidate(self, *, key_id: str | None = None, team_id: str | None = None) -> None:
        if key_id is None and team_id is None:
            self._cache.clear()
            return
        for digest, (_, ctx) in list(self._cache.items()):
            if ctx.key_id == key_id or ctx.team_id == team_id:
                self._cache.pop(digest, None)

    async def authenticate(self, token: str | None) -> AuthContext:
        if not token:
            raise _unauthorized(
                "Missing API key. Pass it as 'Authorization: Bearer <key>'.", "missing"
            )
        if not token.startswith(KEY_PREFIX):
            raise _unauthorized("Invalid API key.", "malformed")
        digest = hash_api_key(token)
        now = time.monotonic()
        cached = self._cache.get(digest)
        if cached is not None and cached[0] > now:
            ctx = cached[1]
        else:
            async with self._session_factory() as session:
                key = await repository.find_api_key_by_hash(session, digest)
                if key is None or not hmac.compare_digest(key.key_hash, digest):
                    raise _unauthorized("Invalid API key.", "unknown")
                if key.revoked_at is not None:
                    raise _unauthorized("This API key has been revoked.", "revoked")
                if not key.team.is_active:
                    raise GatewayError(
                        403, "This team is disabled.", type="permission_error", code="team_disabled"
                    )
                ctx = AuthContext.from_orm(key)
            self._cache[digest] = (now + self._ttl_s, ctx)
        if ctx.expires_at is not None and ctx.expires_at <= datetime.now(UTC):
            raise _unauthorized("This API key has expired.", "expired")
        return ctx


def verify_master_key(expected: str | None, provided: str | None) -> None:
    if expected is None:
        raise GatewayError(
            503,
            "Admin API disabled: set GATEWAY_MASTER_KEY to enable it.",
            type="api_error",
            code="admin_disabled",
        )
    if not provided or not hmac.compare_digest(expected.encode(), provided.encode()):
        raise GatewayError(
            401,
            "Invalid master key.",
            type="authentication_error",
            code="invalid_master_key",
            headers={"WWW-Authenticate": "Bearer"},
        )
