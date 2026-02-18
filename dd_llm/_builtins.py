"""Auto-register built-in adapters on import."""

from __future__ import annotations

import os

from dd_llm.registry import register_adapter
from dd_llm.adapters.claude_cli import ClaudeCLIAdapter
from dd_llm.adapters.openai_sdk import OpenAIAdapter
from dd_llm.adapters.anthropic_sdk import AnthropicAdapter
from dd_llm.adapters.gemini_sdk import GeminiAdapter


def _make_openrouter(**kwargs):
    """Factory for OpenRouter — OpenAI-compatible with custom base_url."""
    return OpenAIAdapter(
        api_key=kwargs.get("api_key") or os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        default_model=kwargs.get("default_model", "anthropic/claude-sonnet-4-5-20250929"),
        provider_name="openrouter",
    )


def _make_ollama(**kwargs):
    """Factory for Ollama — OpenAI-compatible with local base_url."""
    host = kwargs.get("host") or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return OpenAIAdapter(
        api_key="ollama",
        base_url=f"{host}/v1",
        default_model=kwargs.get("default_model", "llama3.2"),
        provider_name="ollama",
    )


register_adapter("claude_cli", ClaudeCLIAdapter)
register_adapter("openai", OpenAIAdapter)
register_adapter("anthropic", AnthropicAdapter)
register_adapter("gemini", GeminiAdapter)
register_adapter("openrouter", _make_openrouter)
register_adapter("ollama", _make_ollama)
