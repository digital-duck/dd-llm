"""LLM adapter abstract base class and response dataclass."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Structured LLM response with metadata.

    Merges fields from PocoFlow's LLMResponse (retry/fallback metadata)
    and SPL's GenerationResult (token usage and cost tracking).
    """

    content: str
    success: bool
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    attempts: int = 1
    total_time: float = 0.0
    error_history: list[dict[str, Any]] | None = None


class LLMAdapter(ABC):
    """Abstract interface for LLM providers.

    All dd-llm backends must implement the synchronous ``call()`` method.
    Adapters that use async SDKs internally should use ``asyncio.run()``
    within their ``call()`` implementation.
    """

    @abstractmethod
    def call(
        self,
        prompt: str = "",
        *,
        model: str = "",
        messages: list[dict] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Synchronous LLM call. Returns LLMResponse (always, even on failure)."""
        ...

    def list_models(self) -> list[str]:
        """List available models for this adapter."""
        return []

    def _measure_time(self) -> float:
        """Return a start time for latency measurement."""
        return time.perf_counter()

    def _elapsed_ms(self, start: float) -> float:
        """Calculate elapsed milliseconds since *start*."""
        return (time.perf_counter() - start) * 1000
