# Custom Provider

Register your own LLM adapter and use it through `call_llm()` and
`UnifiedLLMProvider` — just like any built-in provider.

## Run It

```bash
pip install -r requirements.txt

# Use the echo adapter (no API key needed)
python main.py

# Use the uppercase adapter
python main.py --provider uppercase "Hello World"
```

## What It Shows

- **LLMAdapter subclass**: implement `call()`, return `LLMResponse`
- **register_adapter()**: make it available by name globally
- **call_llm()**: custom providers work with the convenience function
- **UnifiedLLMProvider**: custom providers participate in retry/fallback chains
- **Zero API keys**: great for testing and development
