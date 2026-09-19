"""Application wiring: every long-lived component, built once per process."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from llm_gateway.auth import KeyStore
from llm_gateway.budget import BudgetManager
from llm_gateway.cache import ResponseCache
from llm_gateway.config import GatewayConfig, Settings
from llm_gateway.db import repository
from llm_gateway.db.session import create_engine, create_session_factory
from llm_gateway.db.usage_writer import UsageWriter
from llm_gateway.pricing import PricingTable
from llm_gateway.providers import Provider, build_providers
from llm_gateway.ratelimit import RateLimiter
from llm_gateway.routing.circuit_breaker import CircuitBreakerRegistry
from llm_gateway.routing.router import Router


@dataclass
class GatewayState:
    settings: Settings
    config: GatewayConfig
    redis: Redis
    engine: AsyncEngine
    session_factory: async_sessionmaker
    providers: dict[str, Provider]
    breakers: CircuitBreakerRegistry
    router: Router
    rate_limiter: RateLimiter
    budgets: BudgetManager
    cache: ResponseCache
    pricing: PricingTable
    usage_writer: UsageWriter
    key_store: KeyStore
    background_tasks: set[asyncio.Task[None]] = field(default_factory=set)

    def spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        """Run a coroutine detached from the request (e.g. after a client disconnect)."""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def drain(self) -> None:
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)


def build_state(
    settings: Settings,
    config: GatewayConfig,
    *,
    redis: Redis | None = None,
    providers: dict[str, Provider] | None = None,
) -> GatewayState:
    redis = redis or Redis.from_url(settings.redis_url, decode_responses=True)
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    providers = providers if providers is not None else build_providers(config)
    breakers = CircuitBreakerRegistry(config.circuit_breaker)
    for name in providers:
        breakers.get(name)  # pre-register so every provider exports a state gauge

    async def load_spend(team_id: str, since: datetime) -> float:
        async with session_factory() as session:
            return await repository.sum_team_spend(session, team_id, since)

    return GatewayState(
        settings=settings,
        config=config,
        redis=redis,
        engine=engine,
        session_factory=session_factory,
        providers=providers,
        breakers=breakers,
        router=Router(config, providers, breakers),
        rate_limiter=RateLimiter(redis, fail_open=settings.redis_fail_open),
        budgets=BudgetManager(redis, load_spend, fail_open=settings.redis_fail_open),
        cache=ResponseCache(redis, ttl_s=config.cache.ttl_s, enabled=config.cache.enabled),
        pricing=PricingTable(config.pricing),
        usage_writer=UsageWriter(session_factory, interval_s=settings.usage_flush_interval_s),
        key_store=KeyStore(session_factory, ttl_s=settings.key_cache_ttl_s),
    )
