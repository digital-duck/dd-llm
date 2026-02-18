"""Tests for UnifiedLLMProvider — retry, fallback, stats."""

import pytest

from dd_llm.base import LLMAdapter, LLMResponse
from dd_llm.registry import _ADAPTER_REGISTRY, register_adapter
from dd_llm.provider import UnifiedLLMProvider


class _SuccessAdapter(LLMAdapter):
    def call(self, prompt="", **kwargs):
        return LLMResponse(
            content="ok", success=True, provider="success", model="m"
        )


class _FailAdapter(LLMAdapter):
    def call(self, prompt="", **kwargs):
        raise RuntimeError("always fails")


class _FailThenSucceedAdapter(LLMAdapter):
    def __init__(self):
        self._calls = 0

    def call(self, prompt="", **kwargs):
        self._calls += 1
        if self._calls < 3:
            raise RuntimeError(f"fail #{self._calls}")
        return LLMResponse(
            content="recovered", success=True, provider="flaky", model="m"
        )


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_ADAPTER_REGISTRY)
    yield
    _ADAPTER_REGISTRY.clear()
    _ADAPTER_REGISTRY.update(saved)


class TestUnifiedLLMProvider:
    def test_success(self):
        register_adapter("_test_ok", _SuccessAdapter)
        p = UnifiedLLMProvider(
            primary_provider="_test_ok", fallback_providers=[], max_retries=1
        )
        result = p.call("hello")
        assert result.success
        assert result.content == "ok"

    def test_fallback(self):
        register_adapter("_test_fail", _FailAdapter)
        register_adapter("_test_ok", _SuccessAdapter)
        p = UnifiedLLMProvider(
            primary_provider="_test_fail",
            fallback_providers=["_test_ok"],
            max_retries=1,
            initial_wait=0.01,
        )
        result = p.call("hello")
        assert result.success
        assert result.content == "ok"

    def test_all_fail(self):
        register_adapter("_test_fail", _FailAdapter)
        p = UnifiedLLMProvider(
            primary_provider="_test_fail",
            fallback_providers=[],
            max_retries=2,
            initial_wait=0.01,
        )
        result = p.call("hello")
        assert not result.success
        assert result.provider == "all_failed"
        assert result.error_history

    def test_retry_succeeds(self):
        # Adapter shared across retries — need a factory
        adapter_instance = _FailThenSucceedAdapter()
        register_adapter("_test_flaky", lambda **kw: adapter_instance)
        p = UnifiedLLMProvider(
            primary_provider="_test_flaky",
            fallback_providers=[],
            max_retries=3,
            initial_wait=0.01,
        )
        result = p.call("hello")
        assert result.success
        assert result.content == "recovered"
        assert result.attempts == 3

    def test_provider_override(self):
        register_adapter("_test_ok", _SuccessAdapter)
        register_adapter("_test_fail", _FailAdapter)
        p = UnifiedLLMProvider(
            primary_provider="_test_fail",
            fallback_providers=[],
            max_retries=1,
        )
        # Override provider for this call
        result = p.call("hello", provider="_test_ok")
        assert result.success

    def test_stats_tracking(self):
        register_adapter("_test_ok", _SuccessAdapter)
        p = UnifiedLLMProvider(
            primary_provider="_test_ok", fallback_providers=[], max_retries=1
        )
        p.call("a")
        p.call("b")
        stats = p.get_provider_stats()
        assert "_test_ok" in stats
        assert stats["_test_ok"]["successes"] == 2
        assert stats["_test_ok"]["success_rate"] == 1.0

    def test_requires_prompt_or_messages(self):
        p = UnifiedLLMProvider()
        with pytest.raises(ValueError, match="Either prompt or messages"):
            p.call()

    def test_messages_param(self):
        register_adapter("_test_ok", _SuccessAdapter)
        p = UnifiedLLMProvider(
            primary_provider="_test_ok", fallback_providers=[], max_retries=1
        )
        result = p.call(messages=[{"role": "user", "content": "hi"}])
        assert result.success
