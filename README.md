# dd-llm

Shared LLM abstraction layer for Digital Duck projects.

Zero core dependencies. Adapters lazy-import their SDKs only when used.

## Install

```bash
pip install -e .              # zero deps — claude_cli adapter works out of the box
pip install -e ".[openai]"    # + OpenAI SDK (also covers openrouter, ollama)
pip install -e ".[anthropic]" # + Anthropic SDK
pip install -e ".[gemini]"    # + Google GenAI SDK
pip install -e ".[all]"       # all provider SDKs
```

## Quick Start

```python
from dd_llm import call_llm

# Uses LLM_PROVIDER env var (default: "openai")
response = call_llm("What is 2+2?")

# Specify provider
response = call_llm("Hello", provider="claude_cli")

# With messages
response = call_llm(messages=[
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi"},
])
```

## Built-in Adapters

| Name | Class | SDK | Notes |
|------|-------|-----|-------|
| `claude_cli` | `ClaudeCLIAdapter` | none (subprocess) | Dev provider, $0 cost via Claude Code subscription |
| `openai` | `OpenAIAdapter` | `openai` | Direct OpenAI API |
| `anthropic` | `AnthropicAdapter` | `anthropic` | Direct Anthropic API |
| `gemini` | `GeminiAdapter` | `google-genai` | Direct Google API |
| `openrouter` | `OpenAIAdapter` (configured) | `openai` | OpenAI-compatible endpoint |
| `ollama` | `OpenAIAdapter` (configured) | `openai` | Local OpenAI-compatible endpoint |

## Custom Adapters

```python
from dd_llm import LLMAdapter, LLMResponse, register_adapter, call_llm

class MyAdapter(LLMAdapter):
    def call(self, prompt="", *, messages=None, **kwargs):
        result = my_internal_api(prompt)
        return LLMResponse(content=result, success=True, provider="my_api", model="v1")

register_adapter("my_api", MyAdapter)

# Now usable everywhere
response = call_llm("hello", provider="my_api")
```

## UnifiedLLMProvider

Multi-provider client with retry (exponential backoff + jitter) and automatic
fallback to alternative providers.

```python
from dd_llm import UnifiedLLMProvider

provider = UnifiedLLMProvider(
    primary_provider="openai",
    fallback_providers=["anthropic", "ollama"],
    max_retries=3,
)

result = provider.call("Explain quantum computing")
if result.success:
    print(result.content)
    print(f"Provider: {result.provider}, Model: {result.model}")
    print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Primary provider name | `openai` |
| `LLM_MODEL` | Default model (all providers) | per-provider |
| `LLM_MODEL_OPENAI` | Override model for OpenAI | `gpt-4o` |
| `LLM_MODEL_ANTHROPIC` | Override model for Anthropic | `claude-sonnet-4-5-20250929` |
| `LLM_MODEL_GEMINI` | Override model for Gemini | `gemini-2.0-flash` |
| `LLM_MAX_RETRIES` | Max retries per provider | `3` |
| `LLM_INITIAL_WAIT` | Initial backoff (seconds) | `1` |
| `LLM_MAX_WAIT` | Max backoff (seconds) | `30` |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `OPENROUTER_API_KEY` | OpenRouter API key | — |
| `OLLAMA_HOST` | Ollama base URL | `http://localhost:11434` |

## License

MIT
