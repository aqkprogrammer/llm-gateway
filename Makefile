.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help install dev test lint format check dashboard up down logs loadgen clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (incl. dev) with uv
	uv sync

dev: ## Run the gateway locally with auto-reload (needs Redis on localhost:6379)
	uv run llm-gateway --reload

test: ## Run the test suite (fully offline)
	uv run pytest

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint issues and format
	uv run ruff check --fix .
	uv run ruff format .

check: lint test ## Lint + test (what CI runs)

dashboard: ## Regenerate the Grafana dashboard JSON
	uv run python scripts/build_dashboard.py

up: ## Start the full stack (gateway, redis, prometheus, grafana, jaeger)
	docker compose up -d --build

down: ## Stop the stack and remove volumes
	docker compose down -v

logs: ## Tail gateway logs
	docker compose logs -f gateway

loadgen: ## Drive demo traffic (override: make loadgen ARGS="--duration 300 --chaos")
	uv run python scripts/loadgen.py $(ARGS)

clean: ## Remove caches and local state
	rm -rf .pytest_cache .ruff_cache data
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
