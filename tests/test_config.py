from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_gateway.config import expand_env, load_gateway_config, parse_gateway_config


def test_env_expansion() -> None:
    env = {"A": "1", "EMPTY": ""}
    assert expand_env("${A} ${B:-dflt} ${EMPTY:-x} ${MISSING}", env) == "1 dflt x "


def test_providers_without_keys_are_disabled() -> None:
    cfg = parse_gateway_config(
        "providers:\n  o: {type: openai, api_key: '${OPENAI_API_KEY}'}\n  m: {type: mock}\n",
        environ={},
    )
    assert not cfg.providers["o"].is_usable
    assert cfg.providers["m"].is_usable


def test_unknown_provider_in_route_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown provider"):
        parse_gateway_config(
            "providers:\n  m: {type: mock}\nroutes:\n  r: {targets: [x:y]}\n", environ={}
        )


def test_target_parsing_keeps_colons_in_model() -> None:
    cfg = parse_gateway_config(
        "providers:\n  ollama: {type: ollama}\nroutes:\n  r: {targets: ['ollama:llama3.1:8b']}\n",
        environ={},
    )
    assert cfg.routes["r"].targets[0].model == "llama3.1:8b"


def test_shipped_config_is_valid() -> None:
    cfg = load_gateway_config(Path(__file__).parents[1] / "config" / "gateway.yaml")
    assert cfg.routes["smart"].targets[0].model == "claude-sonnet-5"
    assert cfg.pricing["claude-sonnet-5"].input == 2.0
