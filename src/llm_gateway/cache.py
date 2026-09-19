"""Optional exact-match response cache for deterministic, non-streaming requests.

Only requests with ``temperature == 0`` are cached (anything else is sampled and caching
it would change semantics). Keys are scoped per team so one tenant can never read another
tenant's completions, and include the full canonicalised request body.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.telemetry import metrics

logger = logging.getLogger(__name__)

_EXCLUDED_FIELDS = {"stream", "stream_options", "user"}


def is_cacheable(request: ChatCompletionRequest) -> bool:
    return not request.stream and request.temperature == 0 and (request.n or 1) == 1


def cache_key(team_id: str, route: str, request: ChatCompletionRequest) -> str:
    body = request.model_dump(exclude_none=True, exclude=_EXCLUDED_FIELDS)
    body["model"] = route
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"cache:v1:{team_id}:{digest}"


class ResponseCache:
    def __init__(self, redis: Redis, *, ttl_s: int, enabled: bool = True) -> None:
        self.redis = redis
        self.ttl_s = ttl_s
        self.enabled = enabled

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            raw = await self.redis.get(key)
        except RedisError:
            logger.warning("cache read failed", exc_info=True)
            return None
        metrics.CACHE_REQUESTS.labels("hit" if raw else "miss").inc()
        return json.loads(raw) if raw else None

    async def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            await self.redis.set(key, json.dumps(value), ex=self.ttl_s)
        except RedisError:
            logger.warning("cache write failed", exc_info=True)
