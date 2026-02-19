# dd-llm — Cost Tracking

Demonstrates aggregating **token usage and cost** across multiple LLM calls using `dd-llm`.

## What it shows

- Reading `input_tokens`, `output_tokens`, `cost_usd`, `latency_ms` from `LLMResponse`
- A `CallRecord` dataclass for per-call metrics
- A `CostTracker` helper that accumulates totals and per-provider breakdowns
- Three mock cost profiles simulating real-world pricing tiers
- Running in **demo mode** (no API key) with realistic latency simulation

## Run (no API key required)

```bash
# 6 calls to the default mock-balanced adapter
python main.py

# 10 calls, print every individual call record
python main.py --calls 10 --show-calls

# Round-robin across fast / balanced / premium tiers
python main.py --multi --calls 9 --show-calls
```

## Run with a real provider

```bash
OPENAI_API_KEY=sk-... python main.py --provider openai --calls 5 --show-calls
```

## Key concept

Every `LLMResponse` carries usage metadata:

```python
result = adapter.call(prompt="What is gradient descent?")

print(result.input_tokens)   # tokens sent
print(result.output_tokens)  # tokens received
print(result.cost_usd)       # cost in USD (if provider reports it)
print(result.latency_ms)     # wall-clock latency
```

`CostTracker` accumulates these across calls and reports aggregates:

```python
tracker = CostTracker()
tracker.record(result, prompt_snippet=prompt)
tracker.print_summary()
```

## Mock cost profiles

| Profile          | Input ($/M tokens) | Output ($/M tokens) | Avg latency |
|------------------|--------------------|---------------------|-------------|
| `mock-fast`      | $0.10              | $0.30               | ~80 ms      |
| `mock-balanced`  | $2.50              | $10.00              | ~300 ms     |
| `mock-premium`   | $15.00             | $75.00              | ~800 ms     |

## Example output

```
══════════════════════════════════════════════════════════
  COST & USAGE SUMMARY
══════════════════════════════════════════════════════════
  Total calls       : 9 (9 succeeded)
  Input tokens      : 387
  Output tokens     : 621
  Total tokens      : 1,008
  Total cost        : $0.004851
  Avg latency       : 391 ms

  BY PROVIDER
──────────────────────────────────────────────────────────
  [mock-balanced]
    calls        : 3  (success rate 100%)
    tokens       : 129 in / 207 out
    cost         : $0.002395
    avg latency  : 303 ms
  ...
```
