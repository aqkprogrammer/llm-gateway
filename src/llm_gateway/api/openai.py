"""OpenAI-compatible endpoints: ``/v1/chat/completions`` and ``/v1/models``."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Header
from starlette.responses import Response

from llm_gateway.api.deps import AuthDep, StateDep, current_request_id
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.service import ChatService

router = APIRouter(prefix="/v1", tags=["openai-compatible"])

_CREATED = 1_767_225_600  # 2026-01-01T00:00:00Z; OpenAI clients expect an int here.


@router.post("/chat/completions", summary="Create a chat completion (OpenAI-compatible)")
async def chat_completions(
    body: ChatCompletionRequest,
    state: StateDep,
    auth: AuthDep,
    x_gateway_cache: Annotated[str | None, Header()] = None,
    cache_control: Annotated[str | None, Header()] = None,
) -> Response:
    bypass = (x_gateway_cache or "").lower() == "bypass" or "no-cache" in (cache_control or "")
    return await ChatService(state).complete(
        body, auth, request_id=current_request_id(), bypass_cache=bypass
    )


@router.get("/models", summary="List routable models (aliases and direct targets)")
async def list_models(state: StateDep, auth: AuthDep) -> dict[str, Any]:
    config = state.config
    data: list[dict[str, Any]] = []
    for alias, route in config.routes.items():
        if not auth.can_use(alias):
            continue
        usable = [t for t in route.targets if state.providers[t.provider].enabled]
        data.append(
            {
                "id": alias,
                "object": "model",
                "created": _CREATED,
                "owned_by": "llm-gateway",
                "description": route.description,
                "targets": [str(t) for t in route.targets],
                "available": bool(usable),
            }
        )
    if config.allow_direct_routing:
        for name, provider in state.providers.items():
            if not provider.enabled:
                continue
            for model in provider.config.models:
                model_id = f"{name}:{model}"
                if auth.can_use(model_id):
                    data.append(
                        {
                            "id": model_id,
                            "object": "model",
                            "created": _CREATED,
                            "owned_by": name,
                        }
                    )
    return {"object": "list", "data": data}
