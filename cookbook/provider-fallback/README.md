# Provider Fallback

Demonstrates `UnifiedLLMProvider` — retry with exponential backoff + jitter,
then automatic failover to alternative providers.

This is the core resilience pattern from PocoFlow's `UniversalLLMProvider`,
now available as a standalone dd-llm feature.

## Run It

```bash
pip install -r requirements.txt

# Default: openai → anthropic → ollama
export OPENAI_API_KEY="your-key"
python main.py

# Custom chain: ollama first, then openai fallback
python main.py --primary ollama --fallback openai

# Fast retries for testing
python main.py --retries 1 --initial-wait 0.1
```

## What It Shows

- **Retry with backoff**: exponential backoff + jitter on transient failures
- **Self-healing**: error context injected into retry prompts
- **Provider fallback**: automatic failover when primary is exhausted
- **Per-provider stats**: success rate and average latency tracking
- **LLMResponse metadata**: attempts, total_time, error_history
