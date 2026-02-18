"""Google GenAI (Gemini) SDK adapter."""

from __future__ import annotations

import os

from dd_llm.base import LLMAdapter, LLMResponse


class GeminiAdapter(LLMAdapter):
    """Adapter for the Google Gemini API via google-genai SDK."""

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.default_model = default_model or self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
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

        # Convert messages to a single string for Gemini
        if messages:
            contents = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )
        else:
            contents = prompt

        if system:
            contents = f"System: {system}\n\n{contents}"

        client = self._get_client()
        resp = client.models.generate_content(
            model=effective_model,
            contents=contents,
            **kwargs,
        )

        latency = self._elapsed_ms(start)

        # Extract token usage if available
        input_tokens = 0
        output_tokens = 0
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            input_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0

        return LLMResponse(
            content=resp.text or "",
            success=True,
            provider="gemini",
            model=effective_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
        )

    def list_models(self) -> list[str]:
        return [self.DEFAULT_MODEL]
