# Basic LLM Call

Call any LLM provider with a single function — three different ways.

## Run It

```bash
pip install -r requirements.txt

# Claude CLI (free via subscription, no API key needed)
python main.py --provider claude_cli

# OpenAI
export OPENAI_API_KEY="your-key"
python main.py --provider openai

# Ollama (local)
python main.py --provider ollama --model llama3.2

# Custom prompt
python main.py "Explain gravity in one sentence"
```

## What It Shows

- **call_llm()**: one-liner convenience function — returns a string
- **get_adapter()**: direct adapter access with full LLMResponse metadata
- **Messages format**: OpenAI-style conversation history with system prompt
- **Provider-agnostic**: same code works with any registered adapter
