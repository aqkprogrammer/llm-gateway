"""Drive realistic traffic through the gateway so the Grafana dashboard lights up.

Creates (or reuses) a few demo teams with different budgets and rate limits, then sends
a weighted mix of streaming / non-streaming / cacheable requests across the demo routes.
With ``--chaos`` the ``mock`` provider fails for the middle third of the run: its
circuit breaker opens (then half-opens and closes again on recovery), and routes that
depend on it return 503s - all of it visible on the dashboard.

    uv run python scripts/loadgen.py --duration 120 --concurrency 8
    uv run python scripts/loadgen.py --chaos            # flip failure injection mid-run

Only uses keyless mock routes by default; pass ``--models smart,fast`` to include real
providers (this spends real money).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import httpx

PROMPTS = [
    "Summarise the trade-offs of token bucket vs leaky bucket rate limiting.",
    "Write a haiku about circuit breakers.",
    "Explain exponential backoff with jitter to a new engineer.",
    "What is the capital of Australia?",
    "List three ways to reduce LLM inference cost.",
    "Translate 'resilient infrastructure' into French.",
    "Give me a one-line SQL query that counts rows per day.",
    "Why do distributed systems need idempotency keys?",
]

DEMO_TEAMS: list[dict[str, Any]] = [
    # Generous team: most of the traffic.
    {"name": "search", "monthly_budget_usd": 50.0, "rpm_limit": 600},
    # Tight RPM: produces 429s under load.
    {"name": "growth", "monthly_budget_usd": 20.0, "rpm_limit": 30, "tpm_limit": 20000},
    # Tiny daily budget: hits the soft-limit alert, then 402s.
    {"name": "research", "daily_budget_usd": 0.02, "soft_limit_pct": 0.5},
]


@dataclass
class Stats:
    statuses: Counter[str] = field(default_factory=Counter)
    providers: Counter[str] = field(default_factory=Counter)
    fallbacks: int = 0
    cache_hits: int = 0
    latencies: list[float] = field(default_factory=list)

    def report(self, elapsed: float) -> str:
        total = sum(self.statuses.values())
        lat = sorted(self.latencies) or [0.0]
        p50 = lat[len(lat) // 2]
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        return (
            f"{total} requests in {elapsed:.0f}s ({total / max(elapsed, 1e-9):.1f} req/s) | "
            f"status {dict(sorted(self.statuses.items()))} | "
            f"served by {dict(self.providers.most_common())} | "
            f"fallbacks {self.fallbacks} | cache hits {self.cache_hits} | "
            f"p50 {p50 * 1000:.0f}ms p95 {p95 * 1000:.0f}ms"
        )


async def ensure_keys(client: httpx.AsyncClient, master_key: str) -> list[tuple[str, str]]:
    """Create the demo teams (idempotently) and a fresh key for each."""
    admin = {"Authorization": f"Bearer {master_key}"}
    existing = {t["name"]: t for t in (await client.get("/admin/teams", headers=admin)).json()}
    keys: list[tuple[str, str]] = []
    for spec in DEMO_TEAMS:
        team = existing.get(spec["name"])
        if team is None:
            response = await client.post("/admin/teams", json=spec, headers=admin)
            response.raise_for_status()
            team = response.json()
        response = await client.post(
            "/admin/keys", json={"team_id": team["id"], "name": "loadgen"}, headers=admin
        )
        response.raise_for_status()
        keys.append((spec["name"], response.json()["key"]))
    return keys


async def set_chaos(client: httpx.AsyncClient, master_key: str, provider: str, rate: float) -> None:
    await client.patch(
        f"/admin/providers/{provider}",
        json={"failure_rate": rate},
        headers={"Authorization": f"Bearer {master_key}"},
    )


async def one_request(
    client: httpx.AsyncClient, key: str, model: str, stats: Stats, rng: random.Random
) -> None:
    stream = rng.random() < 0.35
    cacheable = not stream and rng.random() < 0.25
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": rng.choice(PROMPTS)},
        ],
        "max_tokens": rng.choice([32, 64, 128]),
        "stream": stream,
    }
    if cacheable:
        body["temperature"] = 0
    if stream:
        body["stream_options"] = {"include_usage": True}
    headers = {"Authorization": f"Bearer {key}"}
    started = time.perf_counter()
    try:
        if stream:
            async with client.stream(
                "POST", "/v1/chat/completions", json=body, headers=headers
            ) as response:
                async for _ in response.aiter_lines():
                    pass
        else:
            response = await client.post("/v1/chat/completions", json=body, headers=headers)
    except httpx.HTTPError as exc:
        stats.statuses[type(exc).__name__] += 1
        return
    stats.latencies.append(time.perf_counter() - started)
    stats.statuses[str(response.status_code)] += 1
    if response.status_code == 200:
        stats.providers[response.headers.get("x-gateway-provider", "?")] += 1
        stats.fallbacks += int(response.headers.get("x-gateway-fallbacks", "0"))
        stats.cache_hits += response.headers.get("x-gateway-cache") == "hit"


async def worker(
    client: httpx.AsyncClient,
    keys: list[tuple[str, str]],
    models: list[str],
    deadline: float,
    rps: float,
    stats: Stats,
    seed: int,
) -> None:
    rng = random.Random(seed)
    weights = [6, 3, 1][: len(keys)]
    while time.monotonic() < deadline:
        _, key = rng.choices(keys, weights=weights)[0]
        await one_request(client, key, rng.choice(models), stats, rng)
        if rps > 0:
            await asyncio.sleep(rng.expovariate(rps))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base-url", default=os.environ.get("GATEWAY_URL", "http://localhost:8080")
    )
    parser.add_argument(
        "--master-key", default=os.environ.get("GATEWAY_MASTER_KEY", "sk-master-change-me")
    )
    parser.add_argument("--duration", type=float, default=60.0, help="seconds to run")
    parser.add_argument("--concurrency", type=int, default=6, help="parallel workers")
    parser.add_argument("--rps", type=float, default=2.0, help="target req/s per worker (0 = max)")
    parser.add_argument("--models", default="demo,demo-failover,demo-chaos")
    parser.add_argument(
        "--chaos",
        action="store_true",
        help="make the 'mock' provider fail for the middle third of the run",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    stats = Stats()
    async with httpx.AsyncClient(base_url=args.base_url, timeout=60) as client:
        keys = await ensure_keys(client, args.master_key)
        print(f"teams: {', '.join(name for name, _ in keys)} | models: {', '.join(models)}")
        started = time.monotonic()
        deadline = started + args.duration
        workers = [
            asyncio.create_task(
                worker(client, keys, models, deadline, args.rps, stats, args.seed + i)
            )
            for i in range(args.concurrency)
        ]

        async def chaos() -> None:
            await asyncio.sleep(args.duration / 3)
            print(">>> chaos: provider 'mock' now failing 100%")
            await set_chaos(client, args.master_key, "mock", 1.0)
            await asyncio.sleep(args.duration / 3)
            print(">>> chaos: provider 'mock' recovered")
            await set_chaos(client, args.master_key, "mock", 0.0)

        chaos_task = asyncio.create_task(chaos()) if args.chaos else None
        try:
            while not all(w.done() for w in workers):
                await asyncio.sleep(5)
                print(stats.report(time.monotonic() - started))
            await asyncio.gather(*workers)
        finally:
            if chaos_task is not None:
                chaos_task.cancel()
                await set_chaos(client, args.master_key, "mock", 0.0)
        print("done:", stats.report(time.monotonic() - started))


if __name__ == "__main__":
    asyncio.run(main())
