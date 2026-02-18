"""Tests for LLMAdapter ABC and LLMResponse dataclass."""

import pytest

from dd_llm.base import LLMAdapter, LLMResponse


class TestLLMResponse:
    def test_creation_minimal(self):
        r = LLMResponse(content="hello", success=True, provider="test", model="m1")
        assert r.content == "hello"
        assert r.success is True
        assert r.provider == "test"
        assert r.model == "m1"
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.cost_usd is None
        assert r.attempts == 1
        assert r.error_history is None

    def test_creation_full(self):
        r = LLMResponse(
            content="world",
            success=False,
            provider="openai",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=20,
            latency_ms=150.0,
            cost_usd=0.01,
            attempts=3,
            total_time=2.5,
            error_history=[{"error": "timeout"}],
        )
        assert r.input_tokens == 10
        assert r.output_tokens == 20
        assert r.cost_usd == 0.01
        assert r.attempts == 3
        assert len(r.error_history) == 1


class TestLLMAdapterABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            LLMAdapter()

    def test_concrete_implementation(self):
        class DummyAdapter(LLMAdapter):
            def call(self, prompt="", **kwargs):
                return LLMResponse(
                    content=f"echo: {prompt}",
                    success=True,
                    provider="dummy",
                    model="dummy-v1",
                )

        adapter = DummyAdapter()
        result = adapter.call("hi")
        assert result.success
        assert result.content == "echo: hi"

    def test_list_models_default(self):
        class DummyAdapter(LLMAdapter):
            def call(self, prompt="", **kwargs):
                return LLMResponse(content="", success=True, provider="d", model="d")

        assert DummyAdapter().list_models() == []

    def test_timing_helpers(self):
        class DummyAdapter(LLMAdapter):
            def call(self, prompt="", **kwargs):
                return LLMResponse(content="", success=True, provider="d", model="d")

        adapter = DummyAdapter()
        start = adapter._measure_time()
        elapsed = adapter._elapsed_ms(start)
        assert elapsed >= 0
