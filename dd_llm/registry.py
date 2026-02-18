"""Adapter registry and factory.

Provides a global registry so that any code (including end-users) can
register custom LLM adapters and retrieve them by name.
"""

from __future__ import annotations

from typing import Callable

from dd_llm.base import LLMAdapter

# Maps provider name -> class or callable that returns an LLMAdapter
_ADAPTER_REGISTRY: dict[str, type[LLMAdapter] | Callable[..., LLMAdapter]] = {}


def register_adapter(
    name: str, adapter_cls_or_factory: type[LLMAdapter] | Callable[..., LLMAdapter]
) -> None:
    """Register a provider by name.

    Accepts either an ``LLMAdapter`` subclass or a callable that returns one.
    """
    _ADAPTER_REGISTRY[name] = adapter_cls_or_factory


def get_adapter(name: str, **kwargs) -> LLMAdapter:
    """Get an adapter instance by name."""
    if name not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise ValueError(f"Unknown adapter '{name}'. Available: {available}")
    return _ADAPTER_REGISTRY[name](**kwargs)


def list_adapters() -> list[str]:
    """List registered adapter names."""
    return sorted(_ADAPTER_REGISTRY.keys())
