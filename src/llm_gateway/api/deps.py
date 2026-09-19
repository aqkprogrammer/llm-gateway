"""FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from llm_gateway.auth import AuthContext, bearer_token, verify_master_key
from llm_gateway.state import GatewayState
from llm_gateway.telemetry.logging import request_id_var


def get_state(request: Request) -> GatewayState:
    return request.app.state.gateway


StateDep = Annotated[GatewayState, Depends(get_state)]


async def require_api_key(
    state: StateDep,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> AuthContext:
    return await state.key_store.authenticate(bearer_token(authorization, x_api_key))


async def require_master_key(
    state: StateDep,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = state.settings.master_key
    verify_master_key(
        expected.get_secret_value() if expected else None, bearer_token(authorization)
    )


def current_request_id() -> str:
    return request_id_var.get() or "unknown"


AuthDep = Annotated[AuthContext, Depends(require_api_key)]
