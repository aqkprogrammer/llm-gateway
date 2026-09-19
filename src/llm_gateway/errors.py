"""Error types. Everything user-facing is rendered in the OpenAI error envelope."""

from __future__ import annotations

from typing import Any

RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
# Statuses where another provider may well succeed even though retrying this one won't:
# bad credentials, missing model, permission problems on *our* upstream account.
FAILOVER_ONLY_STATUSES = frozenset({401, 402, 403, 404})


class GatewayError(Exception):
    """An error returned to the client in the OpenAI ``{"error": {...}}`` shape."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        self.headers = headers or {}

    def to_body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.param,
                "code": self.code,
            }
        }


class ProviderError(Exception):
    """A failed upstream call, classified for retry / failover / circuit-breaker decisions."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        kind: str = "error",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.kind = kind
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        """Worth retrying against the same provider (after backoff)."""
        if self.kind in ("timeout", "connection"):
            return True
        return self.status_code in RETRYABLE_STATUSES

    @property
    def failover(self) -> bool:
        """Worth trying the next provider in the chain."""
        return (
            self.retryable
            or self.status_code in FAILOVER_ONLY_STATUSES
            or (self.status_code is None)
        )

    @property
    def counts_against_breaker(self) -> bool:
        """Client mistakes (4xx) say nothing about provider health."""
        return self.retryable or self.status_code is None

    def __repr__(self) -> str:
        return (
            f"ProviderError(provider={self.provider!r}, status={self.status_code}, "
            f"kind={self.kind!r}, message={self.message!r})"
        )


class AllTargetsFailedError(Exception):
    """Every target in a route failed or was skipped."""

    def __init__(self, route: str, errors: list[ProviderError]) -> None:
        self.route = route
        self.errors = errors
        detail = "; ".join(f"{e.provider}: {e.message}" for e in errors) or "no usable targets"
        super().__init__(f"all targets for {route!r} failed ({detail})")

    def to_gateway_error(self) -> GatewayError:
        statuses = [e.status_code for e in self.errors if e.status_code is not None]
        # If every upstream said "slow down", say the same thing to the client.
        if statuses and all(s == 429 for s in statuses):
            return GatewayError(
                429,
                str(self),
                type="rate_limit_error",
                code="upstream_rate_limited",
                headers={"Retry-After": "5"},
            )
        return GatewayError(503, str(self), type="api_error", code="all_providers_failed")
