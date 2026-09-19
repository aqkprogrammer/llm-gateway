"""Generate the provisioned Grafana dashboard (deploy/grafana/dashboards/llm-gateway.json).

The dashboard is code: edit this file and run ``make dashboard`` rather than hand-editing
the JSON export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "deploy/grafana/dashboards/llm-gateway.json"
DS = {"type": "prometheus", "uid": "prometheus"}
SEL = 'route=~"$route"'
TEAM = 'team=~"$team"'

_next_id = 0


def _id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


def target(expr: str, legend: str = "", instant: bool = False) -> dict[str, Any]:
    t: dict[str, Any] = {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": "A"}
    if instant:
        t["instant"] = True
        t["range"] = False
    return t


def with_refs(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i, t in enumerate(targets):
        t["refId"] = chr(ord("A") + i)
    return targets


def row(title: str, y: int) -> dict[str, Any]:
    return {
        "type": "row",
        "id": _id(),
        "title": title,
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "panels": [],
    }


def stat(
    title: str,
    expr: str,
    grid: tuple[int, int, int, int],
    unit: str = "short",
    decimals: int | None = None,
    thresholds: list[tuple[str, float | None]] | None = None,
    description: str = "",
) -> dict[str, Any]:
    steps = [{"color": c, "value": v} for c, v in (thresholds or [("green", None)])]
    defaults: dict[str, Any] = {
        "unit": unit,
        "color": {"mode": "thresholds"},
        "thresholds": {"mode": "absolute", "steps": steps},
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    x, y, w, h = grid
    return {
        "type": "stat",
        "id": _id(),
        "title": title,
        "description": description,
        "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [target(expr, instant=True)],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "value",
            "graphMode": "none",
            "justifyMode": "center",
            "textMode": "value",
        },
    }


def timeseries(
    title: str,
    targets: list[dict[str, Any]],
    grid: tuple[int, int, int, int],
    unit: str = "short",
    stack: bool = False,
    description: str = "",
    draw: str = "line",
) -> dict[str, Any]:
    x, y, w, h = grid
    return {
        "type": "timeseries",
        "id": _id(),
        "title": title,
        "description": description,
        "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": with_refs(targets),
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "drawStyle": draw,
                    "lineWidth": 2,
                    "fillOpacity": 18 if stack else 8,
                    "showPoints": "never",
                    "spanNulls": True,
                    "stacking": {"mode": "normal" if stack else "none", "group": "A"},
                },
            },
            "overrides": [],
        },
        "options": {
            "legend": {"displayMode": "table", "placement": "right", "calcs": ["mean", "max"]},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def build() -> dict[str, Any]:
    panels: list[dict[str, Any]] = []
    y = 0

    panels.append(row("Overview", y))
    y += 1
    rate = f"sum(rate(llm_gateway_requests_total{{{SEL}}}[$__rate_interval]))"
    panels += [
        stat("Requests / s", rate, (0, y, 4, 4), unit="reqps", decimals=2),
        stat(
            "Error rate",
            f'sum(rate(llm_gateway_requests_total{{{SEL},status=~"5.."}}[$__rate_interval])) / '
            f"clamp_min({rate}, 1e-9)",
            (4, y, 4, 4),
            unit="percentunit",
            decimals=2,
            thresholds=[("green", None), ("orange", 0.01), ("red", 0.05)],
            description="Share of requests answered with 5xx (all providers failed, etc.).",
        ),
        stat(
            "p95 latency",
            "histogram_quantile(0.95, sum by (le) (rate("
            f"llm_gateway_request_duration_seconds_bucket{{{SEL}}}[$__rate_interval])))",
            (8, y, 4, 4),
            unit="s",
            decimals=2,
            thresholds=[("green", None), ("orange", 5), ("red", 15)],
        ),
        stat(
            "Spend (range)",
            f"sum(increase(llm_gateway_cost_usd_total{{{TEAM}}}[$__range]))",
            (12, y, 4, 4),
            unit="currencyUSD",
            decimals=4,
        ),
        stat(
            "Cache hit ratio",
            'sum(rate(llm_gateway_cache_requests_total{result="hit"}[$__rate_interval])) / '
            "clamp_min(sum(rate(llm_gateway_cache_requests_total[$__rate_interval])), 1e-9)",
            (16, y, 4, 4),
            unit="percentunit",
            decimals=1,
        ),
        stat(
            "Open circuits",
            "count(llm_gateway_circuit_breaker_state == 1) or vector(0)",
            (20, y, 4, 4),
            thresholds=[("green", None), ("red", 1)],
            description="Providers whose circuit breaker is currently open.",
        ),
    ]
    y += 4

    panels.append(row("Traffic & latency", y))
    y += 1
    panels += [
        timeseries(
            "Requests by route and status",
            [
                target(
                    f"sum by (route, status) (rate(llm_gateway_requests_total{{{SEL}}}"
                    "[$__rate_interval]))",
                    "{{route}} {{status}}",
                )
            ],
            (0, y, 12, 8),
            unit="reqps",
            stack=True,
        ),
        timeseries(
            "Requests by provider / model",
            [
                target(
                    f"sum by (provider, model) (rate(llm_gateway_requests_total{{{SEL}}}"
                    "[$__rate_interval]))",
                    "{{provider}} / {{model}}",
                )
            ],
            (12, y, 12, 8),
            unit="reqps",
            stack=True,
        ),
    ]
    y += 8
    panels += [
        timeseries(
            "End-to-end latency by provider (p50 / p95 / p99)",
            [
                target(
                    f"histogram_quantile({q}, sum by (le, provider) (rate("
                    f"llm_gateway_request_duration_seconds_bucket{{{SEL}}}[$__rate_interval])))",
                    f"p{int(q * 100)} {{{{provider}}}}",
                )
                for q in (0.5, 0.95, 0.99)
            ],
            (0, y, 12, 8),
            unit="s",
        ),
        timeseries(
            "Time to first token (p50 / p95)",
            [
                target(
                    f"histogram_quantile({q}, sum by (le, provider, model) (rate("
                    "llm_gateway_time_to_first_token_seconds_bucket[$__rate_interval])))",
                    f"p{int(q * 100)} {{{{provider}}}}/{{{{model}}}}",
                )
                for q in (0.5, 0.95)
            ],
            (12, y, 12, 8),
            unit="s",
            description="Measured for streaming requests from the start of the upstream "
            "attempt to the first delta.",
        ),
    ]
    y += 8

    panels.append(row("Resilience", y))
    y += 1
    panels += [
        timeseries(
            "Fallbacks (from -> to)",
            [
                target(
                    "sum by (route, from_provider, to_provider) (rate(llm_gateway_fallbacks_total"
                    f"{{{SEL}}}[$__rate_interval]))",
                    "{{route}}: {{from_provider}} -> {{to_provider}}",
                )
            ],
            (0, y, 8, 8),
            unit="ops",
        ),
        timeseries(
            "Upstream attempts by outcome",
            [
                target(
                    "sum by (provider, outcome) (rate(llm_gateway_provider_attempts_total"
                    "[$__rate_interval]))",
                    "{{provider}} {{outcome}}",
                )
            ],
            (8, y, 8, 8),
            unit="ops",
            stack=True,
        ),
        timeseries(
            "Retries",
            [
                target(
                    "sum by (provider) (rate(llm_gateway_retries_total[$__rate_interval]))",
                    "{{provider}}",
                )
            ],
            (16, y, 8, 8),
            unit="ops",
        ),
    ]
    y += 8
    panels.append(
        {
            "type": "state-timeline",
            "id": _id(),
            "title": "Circuit breaker state",
            "description": "0 = closed, 1 = open, 2 = half-open",
            "datasource": DS,
            "gridPos": {"h": 7, "w": 16, "x": 0, "y": y},
            "targets": [
                target("max by (provider) (llm_gateway_circuit_breaker_state)", "{{provider}}")
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 1},
                            {"color": "orange", "value": 2},
                        ],
                    },
                    "mappings": [
                        {
                            "type": "value",
                            "options": {
                                "0": {"text": "closed", "color": "green"},
                                "1": {"text": "open", "color": "red"},
                                "2": {"text": "half-open", "color": "orange"},
                            },
                        }
                    ],
                },
                "overrides": [],
            },
            "options": {
                "showValue": "never",
                "mergeValues": True,
                "rowHeight": 0.8,
                "legend": {"displayMode": "list", "placement": "bottom"},
            },
        }
    )
    panels.append(
        timeseries(
            "Circuit transitions",
            [
                target(
                    "sum by (provider, to_state) (increase("
                    "llm_gateway_circuit_breaker_transitions_total[$__rate_interval]))",
                    "{{provider}} -> {{to_state}}",
                )
            ],
            (16, y, 8, 7),
            draw="bars",
        )
    )
    y += 7

    panels.append(row("Tokens & cost", y))
    y += 1
    panels += [
        timeseries(
            "Tokens / s by model",
            [
                target(
                    f"sum by (model, type) (rate(llm_gateway_tokens_total{{{TEAM}}}"
                    "[$__rate_interval]))",
                    "{{model}} {{type}}",
                )
            ],
            (0, y, 8, 8),
            stack=True,
        ),
        timeseries(
            "Spend rate by team (USD / hour)",
            [
                target(
                    f"sum by (team) (rate(llm_gateway_cost_usd_total{{{TEAM}}}[$__rate_interval]))"
                    " * 3600",
                    "{{team}}",
                )
            ],
            (8, y, 8, 8),
            unit="currencyUSD",
            stack=True,
        ),
        {
            "type": "bargauge",
            "id": _id(),
            "title": "Budget utilization (current period)",
            "datasource": DS,
            "gridPos": {"h": 8, "w": 8, "x": 16, "y": y},
            "targets": [
                target(
                    f"max by (team, period) (llm_gateway_budget_utilization_ratio{{{TEAM}}})",
                    "{{team}} ({{period}})",
                    instant=True,
                )
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percentunit",
                    "min": 0,
                    "max": 1,
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "orange", "value": 0.8},
                            {"color": "red", "value": 1},
                        ],
                    },
                },
                "overrides": [],
            },
            "options": {
                "orientation": "horizontal",
                "displayMode": "gradient",
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            },
        },
    ]
    y += 8
    panels.append(
        {
            "type": "table",
            "id": _id(),
            "title": "Spend by team and model (range)",
            "datasource": DS,
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": y},
            "targets": [
                {
                    **target(
                        f"sum by (team, provider, model) (increase(llm_gateway_cost_usd_total"
                        f"{{{TEAM}}}[$__range]))",
                        instant=True,
                    ),
                    "format": "table",
                }
            ],
            "fieldConfig": {
                "defaults": {"unit": "currencyUSD", "decimals": 5},
                "overrides": [],
            },
            "transformations": [
                {"id": "organize", "options": {"excludeByName": {"Time": True}}},
            ],
            "options": {"sortBy": [{"displayName": "Value", "desc": True}]},
        }
    )
    y += 8

    panels.append(row("Guardrails", y))
    y += 1
    panels += [
        timeseries(
            "Rate-limit rejections",
            [
                target(
                    "sum by (team, scope, limit) (rate(llm_gateway_rate_limit_rejections_total"
                    f"{{{TEAM}}}[$__rate_interval]))",
                    "{{team}} {{scope}} {{limit}}",
                )
            ],
            (0, y, 8, 8),
            unit="reqps",
        ),
        timeseries(
            "Budget rejections & soft-limit alerts",
            [
                target(
                    "sum by (team, period) (rate(llm_gateway_budget_rejections_total"
                    f"{{{TEAM}}}[$__rate_interval]))",
                    "rejected {{team}} ({{period}})",
                ),
                target(
                    "sum by (team, period) (increase(llm_gateway_budget_soft_limit_alerts_total"
                    f"{{{TEAM}}}[$__rate_interval]))",
                    "alert {{team}} ({{period}})",
                ),
            ],
            (8, y, 8, 8),
        ),
        timeseries(
            "In-flight requests & auth failures",
            [
                target("sum(llm_gateway_in_flight_requests)", "in flight"),
                target(
                    "sum by (reason) (rate(llm_gateway_auth_failures_total[$__rate_interval]))",
                    "auth failure: {{reason}}",
                ),
            ],
            (16, y, 8, 8),
        ),
    ]

    def variable(name: str, label: str, query: str) -> dict[str, Any]:
        return {
            "name": name,
            "label": label,
            "type": "query",
            "datasource": DS,
            "query": {"query": query, "refId": "PrometheusVariableQueryEditor-VariableQuery"},
            "definition": query,
            "refresh": 2,
            "includeAll": True,
            "multi": True,
            "allValue": ".*",
            "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
            "sort": 1,
        }

    return {
        "uid": "llm-gateway",
        "title": "LLM Gateway",
        "description": "Traffic, latency, resilience, spend and guardrails for the LLM gateway.",
        "tags": ["llm", "gateway"],
        "timezone": "browser",
        "schemaVersion": 41,
        "version": 1,
        "editable": False,
        "graphTooltip": 1,
        "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"},
        "templating": {
            "list": [
                variable("route", "Route", "label_values(llm_gateway_requests_total, route)"),
                variable("team", "Team", "label_values(llm_gateway_tokens_total, team)"),
            ]
        },
        "links": [
            {
                "title": "Traces (Jaeger)",
                "type": "link",
                "url": "/explore?left=%7B%22datasource%22:%22jaeger%22%7D",
                "icon": "external link",
            }
        ],
        "annotations": {"list": []},
        "panels": panels,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT}")


if __name__ == "__main__":
    main()
