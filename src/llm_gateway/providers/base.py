"""Provider interface and shared HTTP plumbing."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from llm_gateway.config import ProviderConfig
from llm_gateway.errors import ProviderError
from llm_gateway.schemas import ChatCompletionRequest, ChatResult, StreamEvent


class Provider(ABC):
    """An upstream LLM API. Implementations translate to/from the OpenAI wire format."""

    default_base_url: str = ""

    def __init__(
        self, name: str, config: ProviderConfig, client: httpx.AsyncClient | None = None
    ) -> None:
        self.name = name
        self.config = config
        self.base_url = (config.base_url or self.default_base_url).rstrip("/")
        self._client = client
        self._owns_client = client is None

    @property
    def type(self) -> str:
        return self.config.type

    @property
    def enabled(self) -> bool:
        return self.config.is_usable

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            limits = httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_connections,
            )
            self._client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(60.0))
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def chat(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> ChatResult:
        """Non-streaming completion."""

    @abstractmethod
    def stream(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion. Must end with an event carrying ``usage`` when known."""

    # -- helpers ---------------------------------------------------------------------------

    def _drop_unsupported(self, payload: dict[str, Any]) -> dict[str, Any]:
        for param in self.config.unsupported_params:
            payload.pop(param, None)
        return payload

    def _error(
        self,
        message: str,
        *,
        status_code: int | None = None,
        kind: str = "error",
        retry_after: float | None = None,
    ) -> ProviderError:
        return ProviderError(
            message, provider=self.name, status_code=status_code, kind=kind, retry_after=retry_after
        )

    def _transport_error(self, exc: httpx.HTTPError) -> ProviderError:
        if isinstance(exc, httpx.TimeoutException):
            return self._error(f"upstream timeout: {type(exc).__name__}", kind="timeout")
        return self._error(f"upstream connection error: {exc!r}", kind="connection")

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        body = (await response.aread()).decode("utf-8", errors="replace")
        raise self._error(
            f"HTTP {response.status_code}: {_extract_error_message(body)}",
            status_code=response.status_code,
            kind=_kind_for_status(response.status_code),
            retry_after=parse_retry_after(response.headers.get("retry-after")),
        )

    def _timeout(self, seconds: float) -> httpx.Timeout:
        return httpx.Timeout(seconds, connect=min(seconds, 10.0))


def _kind_for_status(status: int) -> str:
    if status == 429:
        return "rate_limited"
    if status in (401, 403):
        return "auth_error"
    if status >= 500:
        return "server_error"
    return "client_error"


def _extract_error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except ValueError:
        return body[:500] or "<empty body>"
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
    return body[:500]


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


async def iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str | None, str]]:
    """Yield ``(event, data)`` pairs from a ``text/event-stream`` response."""
    event: str | None = None
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield event, "\n".join(data_lines)
