"""OpenAI Chat Completions (and any OpenAI-compatible server) - mostly pass-through."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.providers.base import Provider, iter_sse
from llm_gateway.schemas import ChatCompletionRequest, ChatResult, StreamEvent, Usage


class OpenAIProvider(Provider):
    default_base_url = "https://api.openai.com/v1"

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", **self.config.headers}
        if self.config.api_key is not None:
            headers["authorization"] = f"Bearer {self.config.api_key.get_secret_value()}"
        return headers

    def build_payload(
        self, request: ChatCompletionRequest, model: str, stream: bool
    ) -> dict[str, Any]:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = model
        payload["stream"] = stream
        payload.pop("stream_options", None)
        if stream:
            # Always ask for usage so the gateway can meter streamed requests.
            payload["stream_options"] = {"include_usage": True}
        return self._drop_unsupported(payload)

    async def chat(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> ChatResult:
        payload = self.build_payload(request, model, stream=False)
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout(timeout),
            )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        await self._raise_for_status(response)
        try:
            result = ChatResult.from_openai(response.json())
        except (ValueError, KeyError, IndexError) as exc:
            raise self._error(f"malformed upstream response: {exc!r}", status_code=502) from exc
        result.model = result.model or model
        return result

    async def stream(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(request, model, stream=True)
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout(timeout),
            ) as response:
                await self._raise_for_status(response)
                async for _event, data in iter_sse(response):
                    if data.strip() == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        error = chunk["error"]
                        message = error.get("message") if isinstance(error, dict) else str(error)
                        raise self._error(f"stream error: {message}", status_code=502)
                    usage = chunk.get("usage")
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        event = StreamEvent(
                            content=delta.get("content"),
                            tool_calls=delta.get("tool_calls") or None,
                            finish_reason=choice.get("finish_reason"),
                            model=chunk.get("model"),
                        )
                        if event.has_delta:
                            yield event
                    if usage:
                        yield StreamEvent(
                            usage=Usage(
                                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                                completion_tokens=int(usage.get("completion_tokens") or 0),
                            ),
                            model=chunk.get("model"),
                        )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
