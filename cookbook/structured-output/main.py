"""dd-llm Structured Output — extract JSON from LLM responses.

Demonstrates:
- Using a system prompt to instruct the LLM to reply in JSON
- Parsing the response with json.loads()
- Validating extracted fields with a simple dataclass
- Graceful fallback when the LLM returns malformed JSON
- Running without a real API key using a built-in mock adapter

Usage:
    python main.py                              # demo mode (mock adapter, no API key)
    python main.py --provider openai            # real OpenAI
    python main.py --provider anthropic         # real Anthropic
    python main.py --provider ollama            # local Ollama
    python main.py --task sentiment             # run sentiment task
    python main.py --task entity                # run entity extraction task
    python main.py --task summary               # run structured summary task
"""

import json
import click
from dataclasses import dataclass
from typing import Any

from dd_llm import (
    LLMAdapter, LLMResponse,
    register_adapter, get_adapter, list_adapters,
)


# ---------------------------------------------------------------------------
# Mock adapter — returns canned JSON without any API key
# ---------------------------------------------------------------------------

# Keys are unique substrings searched in the user prompt (most-specific first).
# "key_points" is unique to the summary task; "entity" is unique to the entity task;
# "confidence" is unique to the sentiment task.
_MOCK_RESPONSES = [
    ("key_points", json.dumps({
        "title": "Quarterly Earnings Beat",
        "key_points": [
            "Revenue up 12% YoY",
            "iPhone sales exceeded expectations",
            "Services segment hit all-time high",
        ],
        "sentiment": "positive",
        "word_count": 42,
    })),
    ("entity_count", json.dumps({
        "entities": [
            {"text": "Apple Inc.", "type": "ORG"},
            {"text": "Tim Cook",   "type": "PERSON"},
            {"text": "Cupertino",  "type": "LOC"},
        ],
        "entity_count": 3,
    })),
    ("confidence", json.dumps({
        "sentiment": "positive",
        "confidence": 0.92,
        "keywords": ["great", "love", "excellent"],
    })),
]

_DEFAULT_MOCK = json.dumps({"result": "ok", "note": "generic mock response"})


class MockStructuredAdapter(LLMAdapter):
    """Returns pre-baked JSON for demo / CI without a real API key."""

    def call(self, prompt="", *, model="", messages=None, system=None,
             max_tokens=4096, temperature=0.7, **kwargs):
        start = self._measure_time()
        history = messages or [{"role": "user", "content": prompt}]
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            prompt,
        )

        # Choose a mock payload based on unique keywords in the user message
        reply = _DEFAULT_MOCK
        for key, payload in _MOCK_RESPONSES:
            if key in last_user.lower():
                reply = payload
                break

        return LLMResponse(
            content=reply,
            success=True,
            provider="mock",
            model="mock-structured-v1",
            input_tokens=sum(len(m["content"].split()) for m in history),
            output_tokens=len(reply.split()),
            latency_ms=self._elapsed_ms(start),
            cost_usd=0.0,
        )


register_adapter("mock", MockStructuredAdapter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise data-extraction assistant. "
    "ALWAYS respond with valid JSON only — no markdown fences, no prose. "
    "Your entire response must be parseable by json.loads()."
)


def call_structured(adapter: LLMAdapter, user_prompt: str, model: str = "") -> dict[str, Any]:
    """Call the LLM and parse the JSON response. Returns a dict (empty on error)."""
    result = adapter.call(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        model=model,
    )

    if not result.success:
        last_err = (result.error_history or [{}])[-1].get("error", "unknown")
        print(f"  [LLM ERROR] {last_err}")
        return {}

    try:
        parsed = json.loads(result.content)
        return parsed
    except json.JSONDecodeError as exc:
        print(f"  [JSON PARSE ERROR] {exc}")
        print(f"  Raw content: {result.content[:200]}")
        return {}


def print_result(label: str, data: dict[str, Any], result_meta: LLMResponse | None = None):
    print(f"\n{'─' * 50}")
    print(f"Task: {label}")
    print(f"{'─' * 50}")
    if data:
        print(json.dumps(data, indent=2))
    else:
        print("  (no structured data extracted)")
    if result_meta:
        print(f"  [{result_meta.input_tokens} in / {result_meta.output_tokens} out"
              f" — {result_meta.latency_ms:.0f}ms]")


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS = {
    "sentiment": {
        "label": "Sentiment Analysis",
        "prompt": (
            "Analyse the sentiment of this review and return JSON with fields: "
            "sentiment (positive/negative/neutral), confidence (0-1 float), "
            "keywords (list of 3 strings).\n\n"
            "Review: 'I absolutely love this product! The quality is great and "
            "delivery was excellent.'"
        ),
    },
    "entity": {
        "label": "Named Entity Extraction",
        "prompt": (
            "Extract named entities from this text. Return JSON with fields: "
            "entities (list of objects each with 'text' and 'type' where type is "
            "ORG/PERSON/LOC/MISC), entity_count (int).\n\n"
            "Text: 'Apple Inc. CEO Tim Cook announced a new product line at the "
            "company's headquarters in Cupertino.'"
        ),
    },
    "summary": {
        "label": "Structured Summary",
        "prompt": (
            "Summarise this news snippet. Return JSON with fields: "
            "title (str), key_points (list of 3 strings), "
            "sentiment (positive/negative/neutral), word_count (int).\n\n"
            "Text: 'Apple reported strong quarterly earnings today, beating analyst "
            "expectations. Revenue grew 12% year-over-year. iPhone sales surpassed "
            "forecasts and the Services segment reached an all-time high.'"
        ),
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--provider", default="mock", help="LLM provider name")
@click.option("--model",    default="",    help="Model override")
@click.option(
    "--task",
    default="all",
    type=click.Choice(["all", "sentiment", "entity", "summary"]),
    help="Which extraction task to run",
)
def main(provider, model, task):
    """Structured output: extract JSON from LLM responses."""

    print(f"Available providers: {list_adapters()}")
    print(f"Using provider: {provider}")
    print(f"\nSystem prompt: {SYSTEM_PROMPT}\n")

    adapter = get_adapter(provider)

    tasks_to_run = list(TASKS.items()) if task == "all" else [(task, TASKS[task])]

    for task_key, task_cfg in tasks_to_run:
        print(f"\n>>> Running task: {task_cfg['label']}")
        print(f"Prompt: {task_cfg['prompt'][:120]}...")

        # Make the raw call so we can access metadata
        result = adapter.call(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": task_cfg["prompt"]},
            ],
            model=model,
        )

        if not result.success:
            last_err = (result.error_history or [{}])[-1].get("error", "unknown")
            print(f"  [LLM ERROR] {last_err}")
            continue

        try:
            parsed = json.loads(result.content)
            print_result(task_cfg["label"], parsed, result)
        except json.JSONDecodeError as exc:
            print(f"  [JSON PARSE ERROR] {exc}")
            print(f"  Raw response (first 300 chars): {result.content[:300]}")

    print(f"\n{'═' * 50}")
    print("Done.")


if __name__ == "__main__":
    main()
