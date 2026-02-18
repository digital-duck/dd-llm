"""UnifiedLLMProvider — retry + fallback chain over registered adapters.

Ported from PocoFlow's UniversalLLMProvider, refactored to use the
adapter registry instead of hard-coded client factories.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

from dd_llm.base import LLMResponse
from dd_llm.registry import get_adapter, list_adapters


class UnifiedLLMProvider:
    """Multi-provider LLM client with self-healing error recovery.

    Self-healing means that when a call fails, the error context is fed back
    into subsequent retry prompts so the LLM can self-correct.  If the
    primary provider is exhausted the client falls back to alternatives.

    Environment variables
    ---------------------
    LLM_PROVIDER        Primary provider name (default: ``"openai"``).
    LLM_MODEL           Default model for all providers.
    LLM_MAX_RETRIES     Max retry attempts per provider (default: 3).
    LLM_INITIAL_WAIT    Initial backoff seconds (default: 1).
    LLM_MAX_WAIT        Maximum backoff seconds (default: 30).
    """

    def __init__(
        self,
        primary_provider: str | None = None,
        fallback_providers: list[str] | None = None,
        max_retries: int | None = None,
        initial_wait: float | None = None,
        max_wait: float | None = None,
    ):
        self.primary_provider = (
            primary_provider or os.environ.get("LLM_PROVIDER", "openai")
        )
        self.fallback_providers = (
            fallback_providers
            if fallback_providers is not None
            else ["anthropic", "gemini", "openrouter", "ollama"]
        )
        self.max_retries = max_retries or int(
            os.environ.get("LLM_MAX_RETRIES", "3")
        )
        self.initial_wait = initial_wait or float(
            os.environ.get("LLM_INITIAL_WAIT", "1")
        )
        self.max_wait = max_wait or float(os.environ.get("LLM_MAX_WAIT", "30"))

        # Per-provider success/failure tracking
        self.provider_stats: dict[str, dict[str, Any]] = {}

    # -- public API ----------------------------------------------------------

    def call(
        self,
        prompt: str | None = None,
        model: str | None = None,
        *,
        messages: list[dict] | None = None,
        provider: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Call the LLM with self-healing retry and provider fallback.

        Parameters
        ----------
        prompt :
            The user prompt (wrapped in a single-message list).
        model :
            Override the default model for this call.
        messages :
            Full conversation history.  When provided, *prompt* is ignored.
        provider :
            Override the primary provider for this call only.
        **kwargs :
            Extra keyword arguments forwarded to the adapter's ``call()``.
        """
        if messages is None and prompt is None:
            raise ValueError("Either prompt or messages must be provided")

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        start_time = time.time()
        error_history: list[dict[str, Any]] = []

        # Build provider order
        primary = provider or self.primary_provider
        providers_to_try = [primary] + [
            p for p in self.fallback_providers if p != primary
        ]

        registered = set(list_adapters())

        for provider_name in providers_to_try:
            if provider_name not in registered:
                continue

            result = self._try_provider(
                provider_name, messages, model, error_history, **kwargs
            )

            if result.success:
                total_time = time.time() - start_time
                result.total_time = total_time
                self._update_stats(provider_name, True, total_time)
                return result

            error_history.extend(result.error_history or [])
            self._update_stats(provider_name, False, time.time() - start_time)

        return LLMResponse(
            content="",
            success=False,
            provider="all_failed",
            model=model or "unknown",
            attempts=len(error_history),
            total_time=time.time() - start_time,
            error_history=error_history,
        )

    def get_provider_stats(self) -> dict[str, Any]:
        """Return per-provider success rates and average response times."""
        return {
            name: {
                **stats,
                "success_rate": stats["successes"]
                / max(stats["successes"] + stats["failures"], 1),
            }
            for name, stats in self.provider_stats.items()
        }

    # -- internals -----------------------------------------------------------

    def _try_provider(
        self,
        provider_name: str,
        messages: list[dict],
        model: str | None,
        global_errors: list[dict[str, Any]],
        **kwargs,
    ) -> LLMResponse:
        """Try a single provider with exponential backoff and error-context injection."""
        try:
            adapter = get_adapter(provider_name)
        except Exception as exc:
            return LLMResponse(
                content="",
                success=False,
                provider=provider_name,
                model=model or "unknown",
                error_history=[{
                    "provider": provider_name,
                    "attempt": 0,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "timestamp": time.time(),
                }],
            )

        wait_time = self.initial_wait
        local_errors: list[dict[str, Any]] = []

        for attempt in range(self.max_retries):
            try:
                # On retries, inject error context so the LLM can self-correct
                effective_messages = (
                    self._add_error_context(messages, local_errors, global_errors)
                    if attempt > 0
                    else messages
                )

                result = adapter.call(
                    messages=effective_messages, model=model or "", **kwargs
                )

                if result.success:
                    result.attempts = attempt + 1
                    result.error_history = local_errors or None
                    return result

                # Adapter returned failure without raising
                local_errors.append({
                    "provider": provider_name,
                    "attempt": attempt + 1,
                    "error": (result.error_history or [{}])[-1].get("error", "unknown"),
                    "error_type": "AdapterFailure",
                    "timestamp": time.time(),
                })

            except Exception as exc:
                local_errors.append({
                    "provider": provider_name,
                    "attempt": attempt + 1,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "timestamp": time.time(),
                })

            if attempt < self.max_retries - 1:
                jitter = random.uniform(0.1, 0.3) * wait_time
                time.sleep(wait_time + jitter)
                wait_time = min(wait_time * 2, self.max_wait)

        return LLMResponse(
            content="",
            success=False,
            provider=provider_name,
            model=model or "unknown",
            attempts=self.max_retries,
            total_time=0.0,
            error_history=local_errors,
        )

    @staticmethod
    def _add_error_context(
        original_messages: list[dict],
        local_errors: list[dict[str, Any]],
        global_errors: list[dict[str, Any]],
    ) -> list[dict]:
        """Inject recent error context so the LLM can self-correct."""
        recent = (local_errors + global_errors)[-3:]
        if not recent:
            return original_messages

        lines = ["Previous attempts failed with the following errors:"]
        for i, err in enumerate(recent, 1):
            lines.append(f"{i}. {err['error_type']}: {err['error']}")
        lines.append("")
        lines.append("Please analyse these errors and provide a corrected response.")

        return original_messages + [{"role": "user", "content": "\n".join(lines)}]

    def _update_stats(self, provider_name: str, success: bool, elapsed: float):
        if provider_name not in self.provider_stats:
            self.provider_stats[provider_name] = {
                "successes": 0,
                "failures": 0,
                "avg_time": 0.0,
            }
        stats = self.provider_stats[provider_name]
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        total = stats["successes"] + stats["failures"]
        stats["avg_time"] = (stats["avg_time"] * (total - 1) + elapsed) / total
