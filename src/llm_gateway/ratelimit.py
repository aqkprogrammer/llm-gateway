"""Distributed token-bucket rate limiting on Redis.

Each limit (requests/min or tokens/min, per key or per team) is a bucket stored as a Redis
hash ``{tokens, ts}``. A single Lua script refills and checks *all* buckets that apply to
a request and only debits them if every one allows it - so a request rejected by the
team TPM bucket does not consume a slot from the key RPM bucket. The script runs
atomically inside Redis, so concurrent gateway replicas cannot race each other.

Token limits are enforced in two phases because the real token count is only known
after the upstream responds:

1. admission: debit an *estimate* (prompt size) from the TPM buckets;
2. settlement: debit (or refund) the difference between actual usage and the estimate.
   Buckets may go negative, which throttles the next requests until they refill.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# KEYS: bucket keys.  ARGV[1]: now in ms ("" = use Redis server TIME).
# ARGV[2..]: (capacity, refill_per_ms, cost) triples, one per key.
# Returns {allowed, limiting_index (1-based, 0 = none), retry_after_ms, remaining...}.
TOKEN_BUCKET_LUA = """
local now
if ARGV[1] == '' then
  local t = redis.call('TIME')
  now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
else
  now = tonumber(ARGV[1])
end

local n = #KEYS
local levels = {}
local allowed = 1
local limiting = 0
local retry_after = 0

for i = 1, n do
  local base = 2 + (i - 1) * 3
  local capacity = tonumber(ARGV[base])
  local rate = tonumber(ARGV[base + 1])
  local cost = tonumber(ARGV[base + 2])
  local state = redis.call('HMGET', KEYS[i], 'tokens', 'ts')
  local tokens = tonumber(state[1])
  local ts = tonumber(state[2])
  if tokens == nil or ts == nil then
    tokens = capacity
    ts = now
  end
  tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
  levels[i] = tokens
  -- A request larger than the whole bucket is admitted when the bucket is full,
  -- otherwise it could never be served.
  local needed = math.min(cost, capacity)
  if tokens < needed then
    allowed = 0
    local wait = math.ceil((needed - tokens) / rate)
    if wait > retry_after then
      retry_after = wait
      limiting = i
    end
  end
end

local result = {allowed, limiting, retry_after}
for i = 1, n do
  local base = 2 + (i - 1) * 3
  local capacity = tonumber(ARGV[base])
  local rate = tonumber(ARGV[base + 1])
  local cost = tonumber(ARGV[base + 2])
  local tokens = levels[i]
  if allowed == 1 then
    tokens = tokens - cost
    redis.call('HSET', KEYS[i], 'tokens', tostring(tokens), 'ts', tostring(now))
    local ttl = math.ceil((capacity - tokens) / rate) + 1000
    redis.call('PEXPIRE', KEYS[i], ttl)
  end
  result[3 + i] = math.floor(tokens)
end
return result
"""

# Adjust a single bucket by ``delta`` tokens without an admission check (settlement).
# KEYS[1]: bucket.  ARGV: now_ms, capacity, refill_per_ms, delta.  Returns new level.
TOKEN_BUCKET_ADJUST_LUA = """
local now
if ARGV[1] == '' then
  local t = redis.call('TIME')
  now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
else
  now = tonumber(ARGV[1])
end
local capacity = tonumber(ARGV[2])
local rate = tonumber(ARGV[3])
local delta = tonumber(ARGV[4])
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now
end
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
tokens = math.min(capacity, tokens - delta)
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('PEXPIRE', KEYS[1], math.ceil((capacity - tokens) / rate) + 1000)
return math.floor(tokens)
"""


@dataclass(frozen=True, slots=True)
class Bucket:
    """A limit of ``limit`` units per minute."""

    key: str
    limit: int
    kind: str  # "requests" | "tokens"
    scope: str  # "key" | "team"

    @property
    def refill_per_ms(self) -> float:
        return self.limit / 60_000.0

    def reset_seconds(self, remaining: float) -> float:
        return max(0.0, (self.limit - remaining) / self.limit * 60.0)


@dataclass(slots=True)
class RateLimitDecision:
    allowed: bool
    headers: dict[str, str]
    retry_after_s: float = 0.0
    limiting: Bucket | None = None


class RateLimiter:
    def __init__(self, redis: Redis, *, fail_open: bool = True, prefix: str = "rl") -> None:
        self.redis = redis
        self.fail_open = fail_open
        self.prefix = prefix
        self._check = redis.register_script(TOKEN_BUCKET_LUA)
        self._adjust = redis.register_script(TOKEN_BUCKET_ADJUST_LUA)

    def buckets(
        self,
        *,
        key_id: str,
        team_id: str,
        key_rpm: int | None,
        key_tpm: int | None,
        team_rpm: int | None,
        team_tpm: int | None,
    ) -> list[Bucket]:
        specs: list[tuple[str, str, str, int | None]] = [
            ("key", key_id, "requests", key_rpm),
            ("team", team_id, "requests", team_rpm),
            ("key", key_id, "tokens", key_tpm),
            ("team", team_id, "tokens", team_tpm),
        ]
        return [
            Bucket(key=f"{self.prefix}:{scope}:{ident}:{kind}", limit=limit, kind=kind, scope=scope)
            for scope, ident, kind, limit in specs
            if limit
        ]

    async def acquire(
        self, buckets: list[Bucket], *, tokens: int, now_ms: int | None = None
    ) -> RateLimitDecision:
        if not buckets:
            return RateLimitDecision(allowed=True, headers={})
        args: list[Any] = ["" if now_ms is None else now_ms]
        for bucket in buckets:
            cost = 1 if bucket.kind == "requests" else max(0, tokens)
            args.extend([bucket.limit, repr(bucket.refill_per_ms), cost])
        try:
            raw = await self._check(keys=[b.key for b in buckets], args=args)
        except RedisError:
            if not self.fail_open:
                raise
            logger.warning("rate limiter unavailable; failing open", exc_info=True)
            return RateLimitDecision(allowed=True, headers={})

        allowed, limiting_index, retry_after_ms = int(raw[0]), int(raw[1]), int(raw[2])
        remaining = [int(v) for v in raw[3:]]
        headers = self._headers(buckets, remaining)
        limiting = buckets[limiting_index - 1] if limiting_index else None
        retry_after_s = retry_after_ms / 1000.0
        if not allowed:
            headers["retry-after"] = str(max(1, math.ceil(retry_after_s)))
        return RateLimitDecision(
            allowed=bool(allowed), headers=headers, retry_after_s=retry_after_s, limiting=limiting
        )

    async def settle(
        self, buckets: list[Bucket], *, delta_tokens: int, now_ms: int | None = None
    ) -> None:
        """Charge (positive) or refund (negative) the estimate error to TPM buckets."""
        if delta_tokens == 0:
            return
        for bucket in buckets:
            if bucket.kind != "tokens":
                continue
            try:
                await self._adjust(
                    keys=[bucket.key],
                    args=[
                        "" if now_ms is None else now_ms,
                        bucket.limit,
                        repr(bucket.refill_per_ms),
                        delta_tokens,
                    ],
                )
            except RedisError:
                if not self.fail_open:
                    raise
                logger.warning("rate limiter settlement failed", exc_info=True)

    @staticmethod
    def _headers(buckets: list[Bucket], remaining: list[int]) -> dict[str, str]:
        """OpenAI-style ``x-ratelimit-*`` headers, reporting the tightest bucket per kind."""
        headers: dict[str, str] = {}
        for kind in ("requests", "tokens"):
            pairs = [(b, r) for b, r in zip(buckets, remaining, strict=True) if b.kind == kind]
            if not pairs:
                continue
            bucket, left = min(pairs, key=lambda p: p[1] / p[0].limit)
            headers[f"x-ratelimit-limit-{kind}"] = str(bucket.limit)
            headers[f"x-ratelimit-remaining-{kind}"] = str(max(0, left))
            headers[f"x-ratelimit-reset-{kind}"] = f"{bucket.reset_seconds(left):.2f}s"
        return headers
