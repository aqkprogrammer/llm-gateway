"""Anthropic Messages API translation (``POST /v1/messages``).

OpenAI -> Anthropic request mapping:

* ``system`` / ``developer`` messages are hoisted into the top-level ``system`` field.
* ``tool`` messages become ``tool_result`` blocks inside a ``user`` turn; assistant
  ``tool_calls`` become ``tool_use`` blocks. Consecutive same-role turns are merged.
* ``image_url`` parts become ``image`` blocks (base64 for data URLs, ``url`` otherwise).
* ``max_tokens`` is required by Anthropic; a configured default is used when absent.
* ``stop`` -> ``stop_sequences``; ``tools``/``tool_choice``/``parallel_tool_calls`` mapped.

Streaming events (``message_start``, ``content_block_*``, ``message_delta``, ...) are
translated to OpenAI ``chat.completion.chunk`` deltas, including tool-call argument
streaming via ``input_json_delta``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.providers.base import Provider, iter_sse
from llm_gateway.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ChatResult,
    StreamEvent,
    Usage,
    message_text,
    new_completion_id,
)

STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

# Anthropic signals overload with 529 inside the stream as an ``error`` event.
_STREAM_ERROR_STATUS = {
    "overloaded_error": 529,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
}


def _convert_image(part: dict[str, Any]) -> dict[str, Any]:
    image = part.get("image_url") or {}
    url = image.get("url", "") if isinstance(image, dict) else str(image)
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header.removeprefix("data:").split(";")[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _convert_content(content: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        kind = part.get("type")
        if kind == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif kind == "image_url":
            blocks.append(_convert_image(part))
    return blocks


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return {"_raw": arguments}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _convert_message(message: ChatMessage) -> tuple[str, list[dict[str, Any]]]:
    if message.role == "tool":
        return "user", [
            {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message_text(message.content),
            }
        ]
    blocks = _convert_content(message.content)
    if message.role == "assistant":
        for call in message.tool_calls or []:
            function = call.get("function") or {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id") or f"toolu_{new_completion_id()[9:]}",
                    "name": function.get("name", ""),
                    "input": _parse_arguments(function.get("arguments")),
                }
            )
        return "assistant", blocks
    return "user", blocks


def _convert_tool_choice(choice: str | dict[str, Any] | None, parallel: bool | None) -> Any:
    result: dict[str, Any] | None = None
    if isinstance(choice, str):
        result = {
            "auto": {"type": "auto"},
            "required": {"type": "any"},
            "none": {"type": "none"},
        }.get(choice)
    elif isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        if name:
            result = {"type": "tool", "name": name}
    if parallel is False:
        result = result or {"type": "auto"}
        if result["type"] != "none":
            result["disable_parallel_tool_use"] = True
    return result


def build_anthropic_payload(
    request: ChatCompletionRequest, model: str, *, default_max_tokens: int, stream: bool
) -> dict[str, Any]:
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role in ("system", "developer"):
            text = message_text(message.content)
            if text:
                system_parts.append(text)
            continue
        role, blocks = _convert_message(message)
        if not blocks:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": request.output_token_limit or default_max_tokens,
    }
    if stream:
        payload["stream"] = True
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if request.temperature is not None:
        payload["temperature"] = min(request.temperature, 1.0)
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop_list:
        payload["stop_sequences"] = request.stop_list
    if request.user:
        payload["metadata"] = {"user_id": request.user}
    if request.tools:
        payload["tools"] = [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": tool["function"].get("parameters")
                or {"type": "object", "properties": {}},
            }
            for tool in request.tools
            if tool.get("type") == "function" and "function" in tool
        ]
    tool_choice = _convert_tool_choice(request.tool_choice, request.parallel_tool_calls)
    if tool_choice is not None and request.tools:
        payload["tool_choice"] = tool_choice
    return payload


def _usage_from(data: dict[str, Any]) -> Usage:
    prompt = (
        int(data.get("input_tokens") or 0)
        + int(data.get("cache_creation_input_tokens") or 0)
        + int(data.get("cache_read_input_tokens") or 0)
    )
    return Usage(prompt_tokens=prompt, completion_tokens=int(data.get("output_tokens") or 0))


def parse_anthropic_response(data: dict[str, Any]) -> ChatResult:
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
    return ChatResult(
        id=f"chatcmpl-{data['id']}" if data.get("id") else new_completion_id(),
        model=data.get("model", ""),
        content="".join(texts) if texts or not tool_calls else None,
        tool_calls=tool_calls or None,
        finish_reason=STOP_REASON_MAP.get(data.get("stop_reason") or "", "stop"),
        usage=_usage_from(data.get("usage") or {}),
    )


class AnthropicProvider(Provider):
    default_base_url = "https://api.anthropic.com"

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "anthropic-version": self.config.anthropic_version,
            **self.config.headers,
        }
        if self.config.api_key is not None:
            headers["x-api-key"] = self.config.api_key.get_secret_value()
        return headers

    def build_payload(
        self, request: ChatCompletionRequest, model: str, stream: bool
    ) -> dict[str, Any]:
        payload = build_anthropic_payload(
            request, model, default_max_tokens=self.config.default_max_tokens, stream=stream
        )
        return self._drop_unsupported(payload)

    async def chat(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> ChatResult:
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/messages",
                json=self.build_payload(request, model, stream=False),
                headers=self._headers(),
                timeout=self._timeout(timeout),
            )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        await self._raise_for_status(response)
        try:
            result = parse_anthropic_response(response.json())
        except (ValueError, KeyError) as exc:
            raise self._error(f"malformed upstream response: {exc!r}", status_code=502) from exc
        result.model = result.model or model
        return result

    async def stream(
        self, request: ChatCompletionRequest, model: str, *, timeout: float
    ) -> AsyncIterator[StreamEvent]:
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=self.build_payload(request, model, stream=True),
                headers=self._headers(),
                timeout=self._timeout(timeout),
            ) as response:
                await self._raise_for_status(response)
                async for event in translate_anthropic_stream(iter_sse(response), self):
                    yield event
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc


async def translate_anthropic_stream(
    events: AsyncIterator[tuple[str | None, str]], provider: Provider
) -> AsyncIterator[StreamEvent]:
    """Translate Anthropic SSE events into provider-neutral :class:`StreamEvent` s."""
    usage = Usage()
    model: str | None = None
    tool_index_by_block: dict[int, int] = {}
    finish_reason: str | None = None

    async for event_name, data in events:
        payload = json.loads(data)
        kind = payload.get("type") or event_name
        if kind == "message_start":
            message = payload.get("message") or {}
            model = message.get("model")
            usage = _usage_from(message.get("usage") or {})
        elif kind == "content_block_start":
            block = payload.get("content_block") or {}
            if block.get("type") == "tool_use":
                tool_index = len(tool_index_by_block)
                tool_index_by_block[payload.get("index", 0)] = tool_index
                yield StreamEvent(
                    tool_calls=[
                        {
                            "index": tool_index,
                            "id": block.get("id"),
                            "type": "function",
                            "function": {"name": block.get("name"), "arguments": ""},
                        }
                    ],
                    model=model,
                )
            elif block.get("type") == "text" and block.get("text"):
                yield StreamEvent(content=block["text"], model=model)
        elif kind == "content_block_delta":
            delta = payload.get("delta") or {}
            if delta.get("type") == "text_delta":
                yield StreamEvent(content=delta.get("text", ""), model=model)
            elif delta.get("type") == "input_json_delta":
                tool_index = tool_index_by_block.get(payload.get("index", 0))
                if tool_index is not None and delta.get("partial_json"):
                    yield StreamEvent(
                        tool_calls=[
                            {
                                "index": tool_index,
                                "function": {"arguments": delta["partial_json"]},
                            }
                        ],
                        model=model,
                    )
            # thinking_delta / signature_delta / citations are not representable in the
            # OpenAI chunk format and are intentionally dropped.
        elif kind == "message_delta":
            delta = payload.get("delta") or {}
            if delta.get("stop_reason"):
                finish_reason = STOP_REASON_MAP.get(delta["stop_reason"], "stop")
            delta_usage = payload.get("usage") or {}
            if "output_tokens" in delta_usage:
                usage.completion_tokens = int(delta_usage["output_tokens"] or 0)
            if delta_usage.get("input_tokens"):
                usage = Usage(
                    prompt_tokens=_usage_from(delta_usage).prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
        elif kind == "message_stop":
            break
        elif kind == "error":
            error = payload.get("error") or {}
            error_type = error.get("type", "api_error")
            raise provider._error(
                f"stream error ({error_type}): {error.get('message', '')}",
                status_code=_STREAM_ERROR_STATUS.get(error_type, 500),
            )
        # "ping" and unknown future event types are ignored.

    yield StreamEvent(finish_reason=finish_reason or "stop", model=model)
    yield StreamEvent(usage=usage, model=model)
