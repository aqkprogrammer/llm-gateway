"""Token -> USD cost from the per-model pricing table (USD per 1M tokens)."""

from __future__ import annotations

import logging

from llm_gateway.config import ModelPrice
from llm_gateway.schemas import Usage

logger = logging.getLogger(__name__)


class PricingTable:
    def __init__(self, prices: dict[str, ModelPrice]) -> None:
        self._prices = prices
        self._warned: set[str] = set()

    def lookup(self, provider: str, model: str) -> ModelPrice | None:
        """``provider:model`` beats ``model``; unknown models are free (and logged once)."""
        price = self._prices.get(f"{provider}:{model}") or self._prices.get(model)
        if price is None and model not in self._warned:
            self._warned.add(model)
            logger.warning(
                "no price configured for model; cost will be reported as 0",
                extra={"provider": provider, "model": model},
            )
        return price

    def cost(self, provider: str, model: str, usage: Usage) -> float:
        price = self.lookup(provider, model)
        if price is None:
            return 0.0
        return (
            usage.prompt_tokens * price.input + usage.completion_tokens * price.output
        ) / 1_000_000
