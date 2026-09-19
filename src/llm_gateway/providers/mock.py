"""Deterministic in-process provider for demos and tests (no network, no API keys).

Responses are a pure function of the prompt and model, so they cache and compare
cleanly. Latency, failure rate, failure status and failure mode are configurable - and
adjustable at runtime through ``PATCH /admin/providers/{name}`` - which makes it easy to
watch retries, failovers and the circuit breaker in Grafana.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections.abc import AsyncIterator

import httpx

from llm_gateway.config import ProviderConfig
from llm_gateway.providers.base import Provider
from llm_gateway.schemas import (
    ChatCompletionRequest,
    ChatResult,
    StreamEvent,
    Usage,
    estimate_prompt_tokens,
    message_text,
)

_VOCABULARY = [
    "gateway",
    "routes",
    "every",
    "request",
    "through",
    "a",
    "resilient",
    "chain",
    "of",
    "providers",
    "while",
    "budgets",
    "rate",
    "limits",
    "and",
    "circuit",
    "breakers",
    "keep",
    "latency",
    "cost",
    "and",
    "failure",
    "blast",
    "radius",
    "under",
    "control",
    "observability",
    "ties",
    "traces",
    "metrics",
    "and",
    "logs",
    "together",
    "so",
    "operators",
    "can",
    "see",
    "exactly",
    "which",
    "model",
    "served",
    "which",
    "team",
    "at",
    "what",
    "price",
]


class MockProvider(Provider):
    def __init__(
        self, name: str, config: ProviderConfig, client: httpx.AsyncClient | None = None
    ) -> None:
        super().__init__(name, config, client)
        self._rng = random.Random(self.config.seed)

    def update(
        self,
        *,
        failure_rate: float | None = None,
        failure_status: int | None = None,
        failure_mode: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        changes: dict[str, object] = {
            k: v
            for k, v in {
                "failure_rate": failure_rate,
                "failure_status": failure_status,
                "failure_mode": failure_mode,
                "latency_ms": latency_ms,
            }.items()
            if v is not None
        }
        self.config = self.config.model_copy(update=changes)

    def _words(self, request: ChatCompletionRequest, model: str) -> list[str]:
        prompt = next(
            (message_text(m.content) for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        digest = hashlib.sha256(f"{model}\x00{prompt}".encode()).digest()
        seeded = random.Random(int.from_bytes(digest[:8], "big"))
        count = self.config.response_words
        words = [seeded.choice(_VOCABULARY) for _ in range(count)]
        prefix = f"[{model}] " + (prompt[:60] + " ->" if prompt else "")
        return [*prefix.split(), *words]

    async def _simulate(self, timeout: float) -> None:
        delay = self.config.latency_ms + self._rng.uniform(0, self.config.jitter_ms)
        await asyncio.sleep(delay / 1000)
        if self.config.failure_rate and self._rng.random() < self.config.failure_rate:
            if self.config.failure_mode == "timeout":
                await asyncio.sleep(timeout + 1)
                raise self._error("mock upstream timed out", kind="timeout")
            status = self.config.failure_status
            raise self._error(
                f"HTTP {status}: mock injected failure",
                status_code=status,
                kind="server_error" if status >= 500 else "client_error",
            )

    def _truncate(self, request: ChatCompletionRequest, words: list[str]) -> tuple[list[str], str]:
        limit = request.output_token_limit
        if limit is not None and len(words) > limit:
            return words[:limit], "length"
        return words, "stop"

    async def chat(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> ChatResult:
        await self._simulate(timeout)
        words, finish = self._truncate(request, self._words(request, model))
        return ChatResult(
            model=model,
            content=" ".join(words),
            finish_reason=finish,
            usage=Usage(
                prompt_tokens=estimate_prompt_tokens(request), completion_tokens=len(words)
            ),
        )

    async def stream(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> AsyncIterator[StreamEvent]:
        await self._simulate(timeout)
        words, finish = self._truncate(request, self._words(request, model))
        for i, word in enumerate(words):
            yield StreamEvent(content=word if i == 0 else f" {word}", model=model)
            if self.config.stream_delay_ms:
                await asyncio.sleep(self.config.stream_delay_ms / 1000)
        yield StreamEvent(finish_reason=finish, model=model)
        yield StreamEvent(
            usage=Usage(
                prompt_tokens=estimate_prompt_tokens(request), completion_tokens=len(words)
            ),
            model=model,
        )
