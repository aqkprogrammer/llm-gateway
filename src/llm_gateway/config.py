"""Runtime settings (environment) and routing configuration (YAML).

Two layers of configuration:

* :class:`Settings` - process-level knobs read from ``GATEWAY_*`` environment variables
  (database/redis URLs, master key, telemetry).
* :class:`GatewayConfig` - the routing document (providers, model aliases, pricing,
  circuit-breaker and cache policy) loaded from YAML. ``${VAR}`` and ``${VAR:-default}``
  references are expanded from the environment before parsing, so secrets never need to
  live in the file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderType = Literal["openai", "anthropic", "ollama", "mock"]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class Settings(BaseSettings):
    """Process settings, read from ``GATEWAY_*`` environment variables (and ``.env``)."""

    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8080
    config_path: Path = Path("config/gateway.yaml")
    database_url: str = "sqlite+aiosqlite:///./data/gateway.db"
    redis_url: str = "redis://localhost:6379/0"
    master_key: SecretStr | None = None

    log_level: str = "INFO"
    log_json: bool = True

    # When Redis is unreachable, keep serving traffic (rate limits / budgets are skipped)
    # instead of failing every request. Availability over strictness by default.
    redis_fail_open: bool = True
    # Authenticated keys are cached in-process for this long; revocations made through the
    # admin API on the same replica take effect immediately, other replicas within the TTL.
    key_cache_ttl_s: float = 10.0
    usage_flush_interval_s: float = 1.0

    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    otel_service_name: str = "llm-gateway"


class ProviderConfig(BaseModel):
    """One upstream provider instance. Type-specific fields are ignored by other types."""

    model_config = ConfigDict(extra="forbid")

    type: ProviderType
    enabled: bool = True
    base_url: str | None = None
    api_key: SecretStr | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    # Request parameters this provider (or its models) rejects; stripped before sending.
    unsupported_params: list[str] = Field(default_factory=list)
    # Models advertised on GET /v1/models when direct routing is enabled.
    models: list[str] = Field(default_factory=list)
    max_connections: int = 100

    # anthropic
    anthropic_version: str = "2023-06-01"
    default_max_tokens: int = 4096

    # mock
    latency_ms: float = 40.0
    jitter_ms: float = 20.0
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    failure_status: int = 503
    failure_mode: Literal["error", "timeout"] = "error"
    response_words: int = 48
    stream_delay_ms: float = 8.0
    seed: int | None = None

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def requires_api_key(self) -> bool:
        return self.type in ("openai", "anthropic")

    @property
    def is_usable(self) -> bool:
        return self.enabled and (not self.requires_api_key or self.api_key is not None)


class RouteTarget(BaseModel):
    """A concrete ``provider:model`` destination."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str

    @classmethod
    def parse(cls, value: str) -> RouteTarget:
        provider, sep, model = value.partition(":")
        if not sep or not provider or not model:
            raise ValueError(f"route target {value!r} must look like 'provider:model'")
        return cls(provider=provider, model=model)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


class RoutePolicy(BaseModel):
    """Timeout and retry policy for a route."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    backoff_base_s: float = Field(default=0.25, ge=0)
    backoff_max_s: float = Field(default=4.0, ge=0)


class RouteConfig(RoutePolicy):
    """A model alias: an ordered fallback chain plus timeout/retry policy."""

    targets: list[RouteTarget] = Field(min_length=1)
    description: str = ""

    @field_validator("targets", mode="before")
    @classmethod
    def _parse_targets(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [RouteTarget.parse(v) if isinstance(v, str) else v for v in value]
        return value


class ModelPrice(BaseModel):
    """USD per one million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = Field(default=5, ge=1)
    recovery_timeout_s: float = Field(default=30.0, gt=0)
    half_open_max_requests: int = Field(default=1, ge=1)
    success_threshold: int = Field(default=1, ge=1)


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: int = Field(default=300, ge=1)


class GatewayConfig(BaseModel):
    """The routing document."""

    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderConfig]
    routes: dict[str, RouteConfig] = Field(default_factory=dict)
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    # Allow clients to address ``provider:model`` directly in addition to aliases.
    allow_direct_routing: bool = True
    # Timeout/retry policy applied to direct ``provider:model`` requests.
    direct_route_defaults: RoutePolicy = Field(default_factory=RoutePolicy)

    @model_validator(mode="after")
    def _check_references(self) -> GatewayConfig:
        for alias, route in self.routes.items():
            if ":" in alias:
                raise ValueError(f"route alias {alias!r} must not contain ':'")
            for target in route.targets:
                if target.provider not in self.providers:
                    raise ValueError(
                        f"route {alias!r} references unknown provider {target.provider!r}"
                    )
        return self

    def direct_route(self, target: RouteTarget) -> RouteConfig:
        return RouteConfig(targets=[target], **self.direct_route_defaults.model_dump())


def expand_env(text: str, environ: dict[str, str] | None = None) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` references."""

    env = os.environ if environ is None else environ

    def _sub(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = env.get(name)
        if value is None or value == "":
            return default if default is not None else ""
        return value

    return _ENV_PATTERN.sub(_sub, text)


def parse_gateway_config(text: str, environ: dict[str, str] | None = None) -> GatewayConfig:
    data = yaml.safe_load(expand_env(text, environ)) or {}
    return GatewayConfig.model_validate(data)


def load_gateway_config(path: Path | str) -> GatewayConfig:
    return parse_gateway_config(Path(path).read_text(encoding="utf-8"))
