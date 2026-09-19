"""OpenAI-compatible request schema and the provider-neutral result types.

The gateway's canonical wire format is OpenAI Chat Completions. Providers translate
*from* :class:`ChatCompletionRequest` and *to* :class:`ChatResult` / :class:`StreamEvent`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions`` body. Unknown fields are kept and passed to OpenAI."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    n: int | None = Field(default=None, ge=1)
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None

    @property
    def output_token_limit(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    @property
    def stop_list(self) -> list[str] | None:
        if self.stop is None:
            return None
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)

    @property
    def include_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.get("include_usage"))


def message_text(content: str | list[dict[str, Any]] | None) -> str:
    """Flatten OpenAI message content (string or parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        str(part.get("text", "")) for part in content if part.get("type") in ("text", None)
    )


def estimate_prompt_tokens(request: ChatCompletionRequest) -> int:
    """Cheap pre-flight token estimate (~4 chars/token) used for TPM admission control."""
    chars = 0
    for message in request.messages:
        chars += len(message_text(message.content))
        for call in message.tool_calls or []:
            chars += len(str(call.get("function", {}).get("arguments", "")))
    return max(1, chars // 4 + 4 * len(request.messages))


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


@dataclass(slots=True)
class ChatResult:
    """A complete (non-streaming) assistant response from any provider."""

    model: str
    content: str | None
    finish_reason: str
    usage: Usage
    tool_calls: list[dict[str, Any]] | None = None
    id: str = field(default_factory=new_completion_id)
    created: int = field(default_factory=lambda: int(time.time()))

    def to_openai(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> ChatResult:
        choice = data["choices"][0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return cls(
            id=data.get("id") or new_completion_id(),
            created=int(data.get("created") or time.time()),
            model=data.get("model", ""),
            content=message.get("content"),
            tool_calls=message.get("tool_calls") or None,
            finish_reason=choice.get("finish_reason") or "stop",
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
            ),
        )


@dataclass(slots=True)
class StreamEvent:
    """One provider-neutral streaming increment.

    ``tool_calls`` uses the OpenAI delta shape (``[{"index": i, "id"?, "function": {...}}]``).
    A usage-only event carries final token counts and produces no client-visible chunk.
    """

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    model: str | None = None

    @property
    def has_delta(self) -> bool:
        return self.content is not None or bool(self.tool_calls) or self.finish_reason is not None
