"""Ollama native chat API translation (``POST /api/chat``, NDJSON streaming)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.providers.base import Provider
from llm_gateway.schemas import (
    ChatCompletionRequest,
    ChatResult,
    StreamEvent,
    Usage,
    message_text,
    new_completion_id,
)


def _images(content: str | list[dict[str, Any]] | None) -> list[str]:
    if not isinstance(content, list):
        return []
    images: list[str] = []
    for part in content:
        if part.get("type") != "image_url":
            continue
        image = part.get("image_url") or {}
        url = image.get("url", "") if isinstance(image, dict) else str(image)
        if url.startswith("data:"):
            images.append(url.partition(",")[2])
    return images


def build_ollama_payload(
    request: ChatCompletionRequest, model: str, *, stream: bool
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        role = "system" if message.role == "developer" else message.role
        converted: dict[str, Any] = {"role": role, "content": message_text(message.content)}
        images = _images(message.content)
        if images:
            converted["images"] = images
        if message.tool_calls:
            converted["tool_calls"] = [
                {
                    "function": {
                        "name": (call.get("function") or {}).get("name", ""),
                        "arguments": _loads_object((call.get("function") or {}).get("arguments")),
                    }
                }
                for call in message.tool_calls
            ]
        if message.role == "tool" and message.name:
            converted["tool_name"] = message.name
        messages.append(converted)

    options: dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.output_token_limit is not None:
        options["num_predict"] = request.output_token_limit
    if request.stop_list:
        options["stop"] = request.stop_list
    if request.seed is not None:
        options["seed"] = request.seed

    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    if options:
        payload["options"] = options
    if request.tools:
        payload["tools"] = request.tools
    response_format = getattr(request, "response_format", None)
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        payload["format"] = "json"
    return payload


def _loads_object(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    return [
        {
            "id": f"call_{new_completion_id()[9:]}",
            "type": "function",
            "function": {
                "name": (call.get("function") or {}).get("name", ""),
                "arguments": json.dumps((call.get("function") or {}).get("arguments") or {}),
            },
        }
        for call in calls
    ]


def _finish_reason(data: dict[str, Any], has_tools: bool) -> str:
    if has_tools:
        return "tool_calls"
    return "length" if data.get("done_reason") == "length" else "stop"


def _usage(data: dict[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=int(data.get("prompt_eval_count") or 0),
        completion_tokens=int(data.get("eval_count") or 0),
    )


def parse_ollama_response(data: dict[str, Any]) -> ChatResult:
    message = data.get("message") or {}
    tool_calls = _tool_calls(message)
    return ChatResult(
        model=data.get("model", ""),
        content=message.get("content", ""),
        tool_calls=tool_calls,
        finish_reason=_finish_reason(data, bool(tool_calls)),
        usage=_usage(data),
    )


class OllamaProvider(Provider):
    default_base_url = "http://localhost:11434"

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", **self.config.headers}
        if self.config.api_key is not None:
            headers["authorization"] = f"Bearer {self.config.api_key.get_secret_value()}"
        return headers

    async def chat(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> ChatResult:
        payload = self._drop_unsupported(build_ollama_payload(request, model, stream=False))
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout(timeout),
            )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        await self._raise_for_status(response)
        try:
            result = parse_ollama_response(response.json())
        except ValueError as exc:
            raise self._error(f"malformed upstream response: {exc!r}", status_code=502) from exc
        result.model = result.model or model
        return result

    async def stream(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> AsyncIterator[StreamEvent]:
        payload = self._drop_unsupported(build_ollama_payload(request, model, stream=True))
        tool_index = 0
        saw_tools = False
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout(timeout),
            ) as response:
                await self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise self._error(f"stream error: {data['error']}", status_code=502)
                    message = data.get("message") or {}
                    calls = _tool_calls(message)
                    if calls:
                        saw_tools = True
                        deltas = []
                        for call in calls:
                            deltas.append({"index": tool_index, **call})
                            tool_index += 1
                        yield StreamEvent(tool_calls=deltas, model=data.get("model"))
                    if message.get("content"):
                        yield StreamEvent(content=message["content"], model=data.get("model"))
                    if data.get("done"):
                        yield StreamEvent(
                            finish_reason=_finish_reason(data, saw_tools), model=data.get("model")
                        )
                        yield StreamEvent(usage=_usage(data), model=data.get("model"))
                        return
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
