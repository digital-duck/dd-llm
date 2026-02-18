"""dd-llm — Shared LLM abstraction layer for Digital Duck projects.

Public API
----------
- ``LLMAdapter``          — Abstract base class for providers
- ``LLMResponse``         — Structured response dataclass
- ``UnifiedLLMProvider``  — Multi-provider client with retry + fallback
- ``register_adapter``    — Register a custom provider
- ``get_adapter``         — Get an adapter instance by name
- ``list_adapters``       — List registered adapter names
- ``call_llm``            — Simple convenience function (returns string)
- ``get_llm_stats``       — Per-provider statistics
"""

from dd_llm.base import LLMAdapter, LLMResponse
from dd_llm.registry import register_adapter, get_adapter, list_adapters
from dd_llm.provider import UnifiedLLMProvider

# Trigger auto-registration of built-in adapters
import dd_llm._builtins  # noqa: F401

__all__ = [
    "LLMAdapter",
    "LLMResponse",
    "UnifiedLLMProvider",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "call_llm",
    "get_llm_stats",
]

# ---------------------------------------------------------------------------
# Convenience functions (lazy singleton)
# ---------------------------------------------------------------------------

_global_llm: UnifiedLLMProvider | None = None


def _get_llm() -> UnifiedLLMProvider:
    global _global_llm
    if _global_llm is None:
        _global_llm = UnifiedLLMProvider()
    return _global_llm


def call_llm(
    prompt: str | None = None,
    *,
    messages: list[dict] | None = None,
    provider: str | None = None,
    **kwargs,
) -> str:
    """Simple LLM call — returns the response text.

    Uses the global :class:`UnifiedLLMProvider` with self-healing retry.
    Pass either *prompt* (single string) or *messages* (conversation list).
    Raises ``RuntimeError`` on failure.
    """
    response = _get_llm().call(prompt, messages=messages, provider=provider, **kwargs)
    if not response.success:
        errors = response.error_history or []
        last = errors[-1]["error"] if errors else "unknown error"
        raise RuntimeError(
            f"LLM call failed after {response.attempts} attempts: {last}"
        )
    return response.content


def get_llm_stats() -> dict:
    """Return per-provider success/failure statistics."""
    return _get_llm().get_provider_stats()
