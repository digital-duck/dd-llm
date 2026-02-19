# dd-llm — Structured Output

Demonstrates extracting **structured JSON** from LLM responses using `dd-llm`.

## What it shows

- Using a **system prompt** to enforce JSON-only replies
- Parsing the response with `json.loads()`
- Three extraction tasks: sentiment analysis, named-entity extraction, structured summary
- Graceful error handling when the LLM returns malformed JSON
- Running in **demo mode** (no API key) via a built-in mock adapter

## Run (no API key required)

```bash
python main.py
```

## Run a specific task

```bash
python main.py --task sentiment
python main.py --task entity
python main.py --task summary
```

## Run with a real provider

```bash
# OpenAI
OPENAI_API_KEY=sk-... python main.py --provider openai --model gpt-4o

# Anthropic
ANTHROPIC_API_KEY=sk-ant-... python main.py --provider anthropic

# Local Ollama (no key needed)
python main.py --provider ollama --model llama3.2
```

## Key concept

The system prompt locks the LLM into JSON-only mode:

```python
SYSTEM_PROMPT = (
    "You are a precise data-extraction assistant. "
    "ALWAYS respond with valid JSON only — no markdown fences, no prose. "
    "Your entire response must be parseable by json.loads()."
)
```

Every call pairs this with a task-specific user prompt:

```python
result = adapter.call(
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
)
parsed = json.loads(result.content)
```

## Example output

```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "keywords": ["great", "love", "excellent"]
}
```
