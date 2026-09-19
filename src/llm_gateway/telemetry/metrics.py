"""Prometheus metrics. All series are prefixed ``llm_gateway_``.

Label cardinality is bounded by configuration: ``route`` is an alias or ``provider:model``
that exists in config, ``team`` is a team name created through the admin API.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, disable_created_metrics

# ``*_created`` timestamp series double the exposition size and are unused by dashboards.
disable_created_metrics()

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)
_TTFT_BUCKETS = (0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15)

REQUESTS = Counter(
    "llm_gateway_requests",
    "Chat completion requests handled by the gateway.",
    ["route", "provider", "model", "status", "stream"],
)
REQUEST_LATENCY = Histogram(
    "llm_gateway_request_duration_seconds",
    "End-to-end gateway latency for chat completions (full body for streams).",
    ["route", "provider", "model", "status"],
    buckets=_LATENCY_BUCKETS,
)
TIME_TO_FIRST_TOKEN = Histogram(
    "llm_gateway_time_to_first_token_seconds",
    "Time from request start until the first streamed delta was received upstream.",
    ["provider", "model"],
    buckets=_TTFT_BUCKETS,
)
PROVIDER_ATTEMPTS = Counter(
    "llm_gateway_provider_attempts",
    "Upstream attempts by outcome (success, error, timeout, circuit_open, ...).",
    ["provider", "model", "outcome"],
)
PROVIDER_LATENCY = Histogram(
    "llm_gateway_provider_attempt_duration_seconds",
    "Latency of individual upstream attempts (until first chunk for streams).",
    ["provider", "model", "outcome"],
    buckets=_LATENCY_BUCKETS,
)
RETRIES = Counter(
    "llm_gateway_retries",
    "Retries against the same provider after a retryable error.",
    ["provider", "model"],
)
FALLBACKS = Counter(
    "llm_gateway_fallbacks",
    "Failovers from one route target to the next.",
    ["route", "from_provider", "to_provider", "reason"],
)
TOKENS = Counter(
    "llm_gateway_tokens",
    "Tokens processed, by direction.",
    ["team", "provider", "model", "type"],
)
COST = Counter(
    "llm_gateway_cost_usd",
    "Spend attributed to teams, in USD.",
    ["team", "provider", "model"],
)
CIRCUIT_STATE = Gauge(
    "llm_gateway_circuit_breaker_state",
    "Circuit breaker state per provider (0=closed, 1=open, 2=half_open).",
    ["provider"],
)
CIRCUIT_TRANSITIONS = Counter(
    "llm_gateway_circuit_breaker_transitions",
    "Circuit breaker state transitions.",
    ["provider", "to_state"],
)
RATE_LIMIT_REJECTIONS = Counter(
    "llm_gateway_rate_limit_rejections",
    "Requests rejected by gateway rate limits.",
    ["team", "scope", "limit"],
)
BUDGET_REJECTIONS = Counter(
    "llm_gateway_budget_rejections",
    "Requests rejected because a team budget was exhausted.",
    ["team", "period"],
)
BUDGET_ALERTS = Counter(
    "llm_gateway_budget_soft_limit_alerts",
    "Soft-limit (alert threshold) crossings per team and budget period.",
    ["team", "period"],
)
BUDGET_UTILIZATION = Gauge(
    "llm_gateway_budget_utilization_ratio",
    "Fraction of the team budget spent in the current period.",
    ["team", "period"],
)
CACHE_REQUESTS = Counter(
    "llm_gateway_cache_requests",
    "Exact-match response cache lookups.",
    ["result"],
)
IN_FLIGHT = Gauge(
    "llm_gateway_in_flight_requests",
    "Chat completion requests currently being processed.",
)
AUTH_FAILURES = Counter(
    "llm_gateway_auth_failures",
    "Rejected API keys.",
    ["reason"],
)
