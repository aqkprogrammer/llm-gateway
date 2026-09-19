from __future__ import annotations

import json

import httpx
import respx

from llm_gateway.config import ProviderConfig
from llm_gateway.providers.ollama import OllamaProvider, build_ollama_payload
from llm_gateway.providers.openai import OpenAIProvider
from llm_gateway.schemas import ChatCompletionRequest

OPENAI = "https://api.openai.test/v1"
OLLAMA = "http://ollama.test:11434"


def req(**fields: object) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "smart", "messages": [{"role": "user", "content": "hi"}], **fields}
    )


@respx.mock
async def test_openai_passthrough_non_streaming() -> None:
    route = respx.post(f"{OPENAI}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-2024-08-06",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )
    )
    p = OpenAIProvider("openai", ProviderConfig(type="openai", api_key="sk-test", base_url=OPENAI))
    result = await p.chat(
        req(temperature=0.3, response_format={"type": "json_object"}), "gpt-4o", timeout=5
    )
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "gpt-4o"
    assert sent["temperature"] == 0.3
    assert sent["response_format"] == {"type": "json_object"}  # extra fields pass through
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-test"
    assert result.content == "Hi!"
    assert result.usage.total_tokens == 5


@respx.mock
async def test_openai_streaming_requests_usage_and_parses_chunks() -> None:
    chunks = [
        {
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        },
        {
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}],
        },
        {
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}],
        },
        {"model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"model": "gpt-4o", "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
    ]
    text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    route = respx.post(f"{OPENAI}/chat/completions").mock(
        return_value=httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
    )
    p = OpenAIProvider("openai", ProviderConfig(type="openai", api_key="k", base_url=OPENAI))
    events = [e async for e in p.stream(req(stream=True), "gpt-4o", timeout=5)]
    sent = json.loads(route.calls.last.request.content)
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert "".join(e.content or "" for e in events) == "Hello"
    assert [e.finish_reason for e in events if e.finish_reason] == ["stop"]
    assert events[-1].usage is not None
    assert events[-1].usage.prompt_tokens == 9


def test_ollama_payload_translation() -> None:
    request = req(
        messages=[
            {"role": "developer", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "f", "content": "ok"},
        ],
        temperature=0,
        max_tokens=50,
        stop=["x"],
        seed=7,
    )
    payload = build_ollama_payload(request, "llama3.1", stream=False)
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["messages"][1]["images"] == ["AAAA"]
    assert payload["messages"][2]["tool_calls"] == [
        {"function": {"name": "f", "arguments": {"a": 1}}}
    ]
    assert payload["messages"][3]["tool_name"] == "f"
    assert payload["options"] == {"temperature": 0, "num_predict": 50, "stop": ["x"], "seed": 7}


@respx.mock
async def test_ollama_non_streaming_and_streaming() -> None:
    p = OllamaProvider("ollama", ProviderConfig(type="ollama", base_url=OLLAMA))
    respx.post(f"{OLLAMA}/api/chat").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "model": "llama3.1",
                    "message": {"role": "assistant", "content": "Hi"},
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 11,
                    "eval_count": 4,
                },
            ),
            httpx.Response(
                200,
                text="\n".join(
                    json.dumps(line)
                    for line in [
                        {"model": "llama3.1", "message": {"content": "He"}, "done": False},
                        {"model": "llama3.1", "message": {"content": "y"}, "done": False},
                        {
                            "model": "llama3.1",
                            "message": {"content": ""},
                            "done": True,
                            "done_reason": "stop",
                            "prompt_eval_count": 5,
                            "eval_count": 2,
                        },
                    ]
                ),
                headers={"content-type": "application/x-ndjson"},
            ),
        ]
    )
    result = await p.chat(req(), "llama3.1", timeout=5)
    assert (result.content, result.finish_reason) == ("Hi", "length")
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (11, 4)

    events = [e async for e in p.stream(req(stream=True), "llama3.1", timeout=5)]
    assert "".join(e.content or "" for e in events) == "Hey"
    assert [e.finish_reason for e in events if e.finish_reason] == ["stop"]
    assert events[-1].usage is not None
    assert events[-1].usage.completion_tokens == 2


@respx.mock
async def test_ollama_tool_calls_become_openai_tool_calls() -> None:
    p = OllamaProvider("ollama", ProviderConfig(type="ollama", base_url=OLLAMA))
    respx.post(f"{OLLAMA}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "weather", "arguments": {"city": "Oslo"}}}
                    ],
                },
                "done": True,
                "done_reason": "stop",
            },
        )
    )
    result = await p.chat(req(), "llama3.1", timeout=5)
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls is not None
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"city": "Oslo"}
