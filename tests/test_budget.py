from __future__ import annotations

from datetime import UTC, datetime

import fakeredis
import pytest

from llm_gateway.budget import BudgetExceededError, BudgetManager, TeamBudget, period_reset
from llm_gateway.config import ModelPrice
from llm_gateway.pricing import PricingTable
from llm_gateway.schemas import Usage

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


def manager(db_spend: float = 0.0) -> tuple[BudgetManager, list[tuple[str, datetime]]]:
    loads: list[tuple[str, datetime]] = []

    async def load(team_id: str, since: datetime) -> float:
        loads.append((team_id, since))
        return db_spend

    redis = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    return BudgetManager(redis, load, clock=lambda: NOW), loads


def budget(daily: float | None = None, monthly: float | None = None) -> TeamBudget:
    return TeamBudget("team_1", "alpha", daily, monthly, soft_limit_pct=0.5)


async def test_spend_accumulates_and_blocks_at_limit() -> None:
    bm, _ = manager()
    b = budget(daily=1.0)
    await bm.check(b)
    await bm.record(b, 0.6)
    await bm.check(b)
    await bm.record(b, 0.4)
    with pytest.raises(BudgetExceededError) as info:
        await bm.check(b)
    assert info.value.period == "daily"
    assert info.value.spent == pytest.approx(1.0)


async def test_monthly_budget_enforced_independently() -> None:
    bm, _ = manager()
    b = budget(daily=100, monthly=0.5)
    await bm.record(b, 0.5)
    with pytest.raises(BudgetExceededError) as info:
        await bm.check(b)
    assert info.value.period == "monthly"


async def test_soft_limit_alert_fires_once_per_period() -> None:
    bm, _ = manager()
    b = budget(daily=1.0)
    assert await bm.record(b, 0.3) == []
    assert await bm.record(b, 0.3) == ["daily"]
    assert await bm.record(b, 0.1) == []


async def test_counters_hydrate_from_database() -> None:
    bm, loads = manager(db_spend=2.5)
    spend = await bm.spend("team_1")
    assert spend == {"daily": 2.5, "monthly": 2.5}
    assert loads[0][1] == datetime(2026, 9, 18, tzinfo=UTC)
    assert loads[1][1] == datetime(2026, 9, 1, tzinfo=UTC)
    await bm.record(budget(daily=10), 0.5)
    assert (await bm.spend("team_1"))["daily"] == pytest.approx(3.0)
    assert len(loads) == 2  # hydrated once


async def test_unlimited_team_is_never_checked() -> None:
    bm, loads = manager()
    await bm.check(budget())
    assert loads == []


def test_period_reset() -> None:
    assert period_reset("daily", NOW) == datetime(2026, 9, 19, tzinfo=UTC)
    assert period_reset("monthly", datetime(2026, 12, 31, 23, tzinfo=UTC)) == datetime(
        2027, 1, 1, tzinfo=UTC
    )


def test_pricing_lookup_and_cost() -> None:
    table = PricingTable(
        {
            "claude-sonnet-5": ModelPrice(input=2, output=10),
            "ollama:claude-sonnet-5": ModelPrice(input=0, output=0),
        }
    )
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
    assert table.cost("anthropic", "claude-sonnet-5", usage) == pytest.approx(7.0)
    assert table.cost("ollama", "claude-sonnet-5", usage) == 0
    assert table.cost("x", "unknown", usage) == 0
