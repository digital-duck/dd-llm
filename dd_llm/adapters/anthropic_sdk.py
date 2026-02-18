"""Anthropic SDK adapter."""

from __future__ import annotations

import os

from dd_llm.base import LLMAdapter, LLMResponse


class AnthropicAdapter(LLMAdapter):
    """Adapter for the Anthropic Messages API."""

    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.default_model = default_model or self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
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
            messages = [{"role": "user", "content": prompt}]

        call_kwargs: dict = {
            "model": effective_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        if system:
            call_kwargs["system"] = system

        client = self._get_client()
        resp = client.messages.create(**call_kwargs)

        latency = self._elapsed_ms(start)
        return LLMResponse(
            content=resp.content[0].text if resp.content else "",
            success=True,
            provider="anthropic",
            model=effective_model,
            input_tokens=resp.usage.input_tokens if resp.usage else 0,
            output_tokens=resp.usage.output_tokens if resp.usage else 0,
            latency_ms=latency,
        )

    def list_models(self) -> list[str]:
        return [self.DEFAULT_MODEL]
