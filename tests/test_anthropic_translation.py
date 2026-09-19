from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_gateway.config import ProviderConfig
from llm_gateway.errors import ProviderError
from llm_gateway.providers.anthropic import AnthropicProvider, build_anthropic_payload
from llm_gateway.schemas import ChatCompletionRequest

BASE = "https://api.anthropic.test"


def provider(**overrides: object) -> AnthropicProvider:
    cfg = ProviderConfig(type="anthropic", api_key="sk-ant-test", base_url=BASE, **overrides)
    return AnthropicProvider("anthropic", cfg)


def req(**fields: object) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({"model": "smart", **fields})


def sse(*events: dict) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def test_request_translation_system_tools_and_images() -> None:
    request = req(
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "developer", "content": "Use metric units."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
                    {"type": "image_url", "image_url": {"url": "https://x.test/cat.png"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "21C"},
            {"role": "user", "content": "Thanks"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        tool_choice="required",
        parallel_tool_calls=False,
        stop="END",
        temperature=1.5,
        max_completion_tokens=256,
        user="u-1",
    )
    payload = build_anthropic_payload(
        request, "claude-sonnet-5", default_max_tokens=4096, stream=True
    )

    assert payload["model"] == "claude-sonnet-5"
    assert payload["system"] == "Be terse.\n\nUse metric units."
    assert payload["max_tokens"] == 256
    assert payload["stream"] is True
    assert payload["temperature"] == 1.0  # clamped to Anthropic's range
    assert payload["stop_sequences"] == ["END"]
    assert payload["metadata"] == {"user_id": "u-1"}
    assert payload["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    assert payload["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"

    roles = [m["role"] for m in payload["messages"]]
    # tool result + following user text merge into one user turn (strict alternation)
    assert roles == ["user", "assistant", "user"]
    user_blocks = payload["messages"][0]["content"]
    assert user_blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
    }
    assert user_blocks[2]["source"] == {"type": "url", "url": "https://x.test/cat.png"}
    assert payload["messages"][1]["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "Paris"}}
    ]
    assert payload["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "21C"},
        {"type": "text", "text": "Thanks"},
    ]


def test_default_max_tokens_and_unsupported_params_dropped() -> None:
    p = provider(unsupported_params=["temperature", "top_p"], default_max_tokens=1234)
    payload = p.build_payload(
        req(messages=[{"role": "user", "content": "hi"}], temperature=0.2, top_p=0.9),
        "claude-sonnet-5",
        stream=False,
    )
    assert payload["max_tokens"] == 1234
    assert "temperature" not in payload
    assert "top_p" not in payload


@respx.mock
async def test_non_streaming_response_translation() -> None:
    route = respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "x"},
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "weather",
                        "input": {"city": "Oslo"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 2,
                    "output_tokens": 7,
                },
            },
        )
    )
    p = provider()
    result = await p.chat(
        req(messages=[{"role": "user", "content": "hi"}]), "claude-sonnet-5", timeout=5
    )

    sent = route.calls.last.request
    assert sent.headers["x-api-key"] == "sk-ant-test"
    assert sent.headers["anthropic-version"] == "2023-06-01"
    body = result.to_openai()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["choices"][0]["message"]["content"] == "Checking."
    call = body["choices"][0]["message"]["tool_calls"][0]
    assert call["id"] == "toolu_1"
    assert json.loads(call["function"]["arguments"]) == {"city": "Oslo"}
    assert body["usage"] == {"prompt_tokens": 17, "completion_tokens": 7, "total_tokens": 24}
    await p.aclose()


@respx.mock
async def test_streaming_translation_text_and_tool_calls() -> None:
    body = sse(
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "model": "claude-sonnet-5",
                "usage": {"input_tokens": 25, "output_tokens": 1},
            },
        },
        {"type": "ping"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_9", "name": "weather", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": ' "Rome"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 42},
        },
        {"type": "message_stop"},
    )
    respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )
    p = provider()
    events = [
        e
        async for e in p.stream(
            req(messages=[{"role": "user", "content": "hi"}]), "claude-sonnet-5", timeout=5
        )
    ]

    text = "".join(e.content or "" for e in events)
    assert text == "Hello"
    tool_events = [e.tool_calls[0] for e in events if e.tool_calls]
    assert tool_events[0] == {
        "index": 0,
        "id": "toolu_9",
        "type": "function",
        "function": {"name": "weather", "arguments": ""},
    }
    args = "".join(t["function"]["arguments"] for t in tool_events)
    assert json.loads(args) == {"city": "Rome"}
    assert [e.finish_reason for e in events if e.finish_reason] == ["tool_calls"]
    usage = events[-1].usage
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens) == (25, 42)


@respx.mock
async def test_stream_error_event_is_retryable_overload() -> None:
    body = sse(
        {"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 1}}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )
    with pytest.raises(ProviderError) as info:
        async for _ in provider().stream(
            req(messages=[{"role": "user", "content": "x"}]), "m", timeout=5
        ):
            pass
    assert info.value.status_code == 529
    assert info.value.retryable


@respx.mock
@pytest.mark.parametrize(
    ("status", "retryable", "failover"),
    [
        (429, True, True),
        (529, True, True),
        (500, True, True),
        (401, False, True),
        (400, False, False),
    ],
)
async def test_http_errors_are_classified(status: int, retryable: bool, failover: bool) -> None:
    respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(
            status,
            json={"type": "error", "error": {"type": "x", "message": "nope"}},
            headers={"retry-after": "2"},
        )
    )
    with pytest.raises(ProviderError) as info:
        await provider().chat(req(messages=[{"role": "user", "content": "x"}]), "m", timeout=5)
    err = info.value
    assert err.status_code == status
    assert "nope" in err.message
    assert err.retryable is retryable
    assert err.failover is failover
    assert err.retry_after == 2.0


@respx.mock
async def test_timeout_maps_to_retryable_error() -> None:
    respx.post(f"{BASE}/v1/messages").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderError) as info:
        await provider().chat(req(messages=[{"role": "user", "content": "x"}]), "m", timeout=5)
    assert info.value.kind == "timeout"
    assert info.value.retryable
