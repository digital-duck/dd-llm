# dd-llm — Multi-Turn Chat

Demonstrates persistent conversation history with `dd-llm`.

## What it shows

- Building a `messages` list across turns
- Passing the full history to every `adapter.call()` so the LLM has context
- Setting a **system prompt** to define the assistant's persona
- Accumulating **token usage** across turns for cost estimation
- A `ChatSession` helper class that wraps all the above
- Running in **demo mode** (no API key) using a built-in mock adapter
- Running in **interactive mode** with a real provider

## Run (no API key required)

```bash
python main.py
```

## Run with a real provider

```bash
# OpenAI
OPENAI_API_KEY=sk-... python main.py --provider openai --model gpt-4o

# Anthropic
ANTHROPIC_API_KEY=sk-ant-... python main.py --provider anthropic

# Local Ollama (no key needed)
python main.py --provider ollama --model llama3.2

# Interactive mode
OPENAI_API_KEY=sk-... python main.py --provider openai --interactive
```

## Key concept

Every call passes the **entire history** as `messages=`:

```python
history.append({"role": "user", "content": user_message})
result = adapter.call(messages=history)
history.append({"role": "assistant", "content": result.content})
```

This gives the LLM full context of the conversation on each turn.
