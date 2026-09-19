"""Upstream provider adapters."""

from __future__ import annotations

from llm_gateway.config import GatewayConfig, ProviderConfig
from llm_gateway.providers.anthropic import AnthropicProvider
from llm_gateway.providers.base import Provider
from llm_gateway.providers.mock import MockProvider
from llm_gateway.providers.ollama import OllamaProvider
from llm_gateway.providers.openai import OpenAIProvider

PROVIDER_CLASSES: dict[str, type[Provider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "mock": MockProvider,
}


def build_provider(name: str, config: ProviderConfig) -> Provider:
    return PROVIDER_CLASSES[config.type](name, config)


def build_providers(config: GatewayConfig) -> dict[str, Provider]:
    return {name: build_provider(name, cfg) for name, cfg in config.providers.items()}


__all__ = [
    "PROVIDER_CLASSES",
    "AnthropicProvider",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "Provider",
    "build_provider",
    "build_providers",
]
