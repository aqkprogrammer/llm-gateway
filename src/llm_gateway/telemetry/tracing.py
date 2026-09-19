"""OpenTelemetry tracing setup.

Span hierarchy for a chat completion::

    POST /v1/chat/completions          (FastAPI server span, auto-instrumented)
    └── gateway.request                (team, route, cost, tokens, cache hit, fallbacks)
        └── gateway.routing            (route, targets, fallback count, served provider)
            ├── gateway.provider_attempt   (provider, model, attempt #, outcome)
            │   └── POST https://...        (httpx client span, auto-instrumented)
            └── gateway.provider_attempt   (next target after a failover)
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from llm_gateway import __version__
from llm_gateway.config import Settings

logger = logging.getLogger(__name__)

tracer = trace.get_tracer("llm_gateway", __version__)

_configured: TracerProvider | None = None


def setup_tracing(settings: Settings) -> TracerProvider | None:
    """Install a global tracer provider exporting OTLP/HTTP. Idempotent per process."""
    global _configured
    if not settings.otel_enabled:
        return None
    if _configured is not None:
        return _configured

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    resource = Resource.create(
        {"service.name": settings.otel_service_name, "service.version": __version__}
    )
    provider = TracerProvider(resource=resource)
    endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/")
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    HTTPXClientInstrumentor().instrument()
    _configured = provider
    logger.info("tracing enabled", extra={"otlp_endpoint": endpoint})
    return provider


def shutdown_tracing() -> None:
    if _configured is not None:
        _configured.shutdown()
