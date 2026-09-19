"""Per-team spend tracking and budget enforcement.

Hot path: spend counters live in Redis (``INCRBYFLOAT`` on per-day and per-month keys,
UTC). The durable source of truth is the ``usage_records`` table; when a counter key is
missing (Redis restart, new period, eviction) it is re-hydrated from the database with
``SET NX`` before use, so a Redis flush never silently resets a team's budget.

Enforcement is pre-flight: a request is rejected once spend has *reached* the limit.
Requests already in flight when the limit is crossed complete normally, so a team can
overshoot by at most its concurrent in-flight cost - the usual trade-off for not
reserving worst-case cost up front.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from redis.asyncio import Redis
from redis.exceptions import RedisError

from llm_gateway.telemetry import metrics

logger = logging.getLogger(__name__)

SpendLoader = Callable[[str, datetime], Awaitable[float]]

PERIODS = ("daily", "monthly")


@dataclass(frozen=True, slots=True)
class TeamBudget:
    team_id: str
    team_name: str
    daily_usd: float | None
    monthly_usd: float | None
    soft_limit_pct: float

    def limit(self, period: str) -> float | None:
        return self.daily_usd if period == "daily" else self.monthly_usd


class BudgetExceededError(Exception):
    def __init__(self, period: str, limit: float, spent: float) -> None:
        super().__init__(f"{period} budget of ${limit:.4f} exhausted (spent ${spent:.4f})")
        self.period = period
        self.limit = limit
        self.spent = spent


def period_start(period: str, now: datetime) -> datetime:
    now = now.astimezone(UTC)
    if period == "daily":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def period_reset(period: str, now: datetime) -> datetime:
    start = period_start(period, now)
    if period == "daily":
        return start + timedelta(days=1)
    return (start + timedelta(days=32)).replace(day=1)


class BudgetManager:
    def __init__(
        self,
        redis: Redis,
        load_spend: SpendLoader,
        *,
        fail_open: bool = True,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.redis = redis
        self._load_spend = load_spend
        self.fail_open = fail_open
        self._clock = clock

    def _key(self, team_id: str, period: str, now: datetime) -> str:
        stamp = now.strftime("%Y-%m-%d") if period == "daily" else now.strftime("%Y-%m")
        return f"spend:{team_id}:{period}:{stamp}"

    async def _ensure(self, team_id: str, period: str, now: datetime) -> str:
        key = self._key(team_id, period, now)
        if not await self.redis.exists(key):
            spent = await self._load_spend(team_id, period_start(period, now))
            ttl = int((period_reset(period, now) - now).total_seconds()) + 86_400
            await self.redis.set(key, repr(float(spent)), ex=ttl, nx=True)
        return key

    async def spend(self, team_id: str) -> dict[str, float]:
        now = self._clock()
        result: dict[str, float] = {}
        for period in PERIODS:
            key = await self._ensure(team_id, period, now)
            result[period] = float(await self.redis.get(key) or 0.0)
        return result

    async def check(self, budget: TeamBudget) -> None:
        """Raise :class:`BudgetExceededError` if any configured budget is exhausted."""
        if budget.daily_usd is None and budget.monthly_usd is None:
            return
        try:
            spent = await self.spend(budget.team_id)
        except RedisError:
            if not self.fail_open:
                raise
            logger.warning("budget store unavailable; failing open", exc_info=True)
            return
        for period in PERIODS:
            limit = budget.limit(period)
            if limit is not None and spent[period] >= limit:
                metrics.BUDGET_REJECTIONS.labels(budget.team_name, period).inc()
                raise BudgetExceededError(period, limit, spent[period])

    async def record(self, budget: TeamBudget, cost_usd: float) -> list[str]:
        """Add spend; return the periods whose soft-limit alert fired on this request."""
        if cost_usd <= 0:
            return []
        now = self._clock()
        fired: list[str] = []
        try:
            for period in PERIODS:
                key = await self._ensure(budget.team_id, period, now)
                total = float(await self.redis.incrbyfloat(key, cost_usd))
                limit = budget.limit(period)
                if not limit:
                    continue
                metrics.BUDGET_UTILIZATION.labels(budget.team_name, period).set(total / limit)
                if total >= limit * budget.soft_limit_pct:
                    alert_key = f"budget-alert:{key}"
                    ttl = int((period_reset(period, now) - now).total_seconds()) + 60
                    if await self.redis.set(alert_key, "1", ex=ttl, nx=True):
                        fired.append(period)
                        metrics.BUDGET_ALERTS.labels(budget.team_name, period).inc()
                        logger.warning(
                            "team budget soft limit reached",
                            extra={
                                "team": budget.team_name,
                                "team_id": budget.team_id,
                                "period": period,
                                "spent_usd": round(total, 6),
                                "limit_usd": limit,
                                "soft_limit_pct": budget.soft_limit_pct,
                            },
                        )
        except RedisError:
            if not self.fail_open:
                raise
            logger.warning("failed to record spend in redis", exc_info=True)
        return fired
