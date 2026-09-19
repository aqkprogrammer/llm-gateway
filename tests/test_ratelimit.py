from __future__ import annotations

import fakeredis

from llm_gateway.ratelimit import RateLimiter


def limiter() -> RateLimiter:
    return RateLimiter(
        fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    )


async def test_request_bucket_allows_capacity_then_rejects_with_retry_after() -> None:
    rl = limiter()
    buckets = rl.buckets(
        key_id="k", team_id="t", key_rpm=3, key_tpm=None, team_rpm=None, team_tpm=None
    )
    now = 1_000_000
    for expected_remaining in (2, 1, 0):
        d = await rl.acquire(buckets, tokens=10, now_ms=now)
        assert d.allowed
        assert d.headers["x-ratelimit-remaining-requests"] == str(expected_remaining)
        assert d.headers["x-ratelimit-limit-requests"] == "3"
    d = await rl.acquire(buckets, tokens=10, now_ms=now)
    assert not d.allowed
    assert d.limiting is not None
    assert d.limiting.kind == "requests"
    # 3 req/min -> one token every 20s
    assert 19.9 < d.retry_after_s <= 20.0
    assert d.headers["retry-after"] == "20"


async def test_bucket_refills_over_time() -> None:
    rl = limiter()
    buckets = rl.buckets(
        key_id="k", team_id="t", key_rpm=60, key_tpm=None, team_rpm=None, team_tpm=None
    )
    now = 5_000_000
    for _ in range(60):
        assert (await rl.acquire(buckets, tokens=1, now_ms=now)).allowed
    assert not (await rl.acquire(buckets, tokens=1, now_ms=now)).allowed
    assert (await rl.acquire(buckets, tokens=1, now_ms=now + 1000)).allowed  # 1 token/s


async def test_multi_bucket_is_atomic_all_or_nothing() -> None:
    rl = limiter()
    buckets = rl.buckets(
        key_id="k", team_id="t", key_rpm=100, key_tpm=None, team_rpm=None, team_tpm=50
    )
    now = 1_000
    ok = await rl.acquire(buckets, tokens=40, now_ms=now)
    assert ok.allowed
    denied = await rl.acquire(buckets, tokens=40, now_ms=now)
    assert not denied.allowed
    assert denied.limiting is not None
    assert (denied.limiting.scope, denied.limiting.kind) == (
        "team",
        "tokens",
    )
    # The rejected request must not have consumed a key RPM slot.
    ok2 = await rl.acquire(buckets, tokens=1, now_ms=now)
    assert ok2.allowed
    assert ok2.headers["x-ratelimit-remaining-requests"] == "98"


async def test_oversized_request_admitted_when_full_and_settlement_goes_negative() -> None:
    rl = limiter()
    buckets = rl.buckets(
        key_id="k", team_id="t", key_rpm=None, key_tpm=100, team_rpm=None, team_tpm=None
    )
    now = 10_000
    assert (await rl.acquire(buckets, tokens=500, now_ms=now)).allowed  # larger than bucket
    d = await rl.acquire(buckets, tokens=1, now_ms=now)
    assert not d.allowed
    assert d.headers["x-ratelimit-remaining-tokens"] == "0"


async def test_settle_refunds_and_charges() -> None:
    rl = limiter()
    buckets = rl.buckets(
        key_id="k", team_id="t", key_rpm=None, key_tpm=1000, team_rpm=None, team_tpm=None
    )
    now = 10_000
    await rl.acquire(buckets, tokens=600, now_ms=now)
    await rl.settle(buckets, delta_tokens=-500, now_ms=now)  # actual usage was 100
    d = await rl.acquire(buckets, tokens=800, now_ms=now)
    assert d.allowed  # 900 available after refund
    await rl.settle(buckets, delta_tokens=300, now_ms=now)  # under-estimated
    d = await rl.acquire(buckets, tokens=1, now_ms=now)
    assert not d.allowed


async def test_no_limits_configured_is_noop() -> None:
    rl = limiter()
    d = await rl.acquire([], tokens=10**9)
    assert d.allowed
    assert d.headers == {}


async def test_uses_redis_server_time_when_now_not_given() -> None:
    rl = limiter()
    buckets = rl.buckets(
        key_id="k", team_id="t", key_rpm=2, key_tpm=None, team_rpm=None, team_tpm=None
    )
    assert (await rl.acquire(buckets, tokens=1)).allowed
    assert (await rl.acquire(buckets, tokens=1)).allowed
    assert not (await rl.acquire(buckets, tokens=1)).allowed
