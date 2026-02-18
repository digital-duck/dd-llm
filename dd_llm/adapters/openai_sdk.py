"""OpenAI SDK adapter — also covers OpenRouter and Ollama via base_url."""

from __future__ import annotations

import os

from dd_llm.base import LLMAdapter, LLMResponse


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI-compatible APIs (OpenAI, OpenRouter, Ollama).

    Parameters
    ----------
    api_key : str or None
        API key.  Falls back to ``OPENAI_API_KEY`` env var.
    base_url : str or None
        Custom base URL for OpenAI-compatible endpoints.
    default_model : str
        Default model when none is specified per-call.
    provider_name : str
        Name used in LLMResponse.provider (e.g. "openai", "openrouter").
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "gpt-4o",
        provider_name: str = "openai",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.default_model = default_model
        self.provider_name = provider_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs: dict = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

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
        start = self._measure_time()
        effective_model = model or self.default_model

        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        elif system:
            messages = [{"role": "system", "content": system}] + messages

        client = self._get_client()
        resp = client.chat.completions.create(
            model=effective_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        latency = self._elapsed_ms(start)
        usage = resp.usage
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            success=True,
            provider=self.provider_name,
            model=effective_model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency,
        )

    def list_models(self) -> list[str]:
        try:
            client = self._get_client()
            return [m.id for m in client.models.list().data]
        except Exception:
            return []
