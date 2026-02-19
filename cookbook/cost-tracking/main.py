"""dd-llm Cost Tracking — aggregate token usage and cost across calls.

Demonstrates:
- Collecting input_tokens, output_tokens, cost_usd, latency_ms from LLMResponse
- Building a CallRecord dataclass to store per-call metrics
- A CostTracker helper that accumulates and reports stats across providers
- Simulating calls to multiple "providers" with different cost profiles
- Running without a real API key using configurable mock adapters

Usage:
    python main.py                              # demo (mock, no API key)
    python main.py --provider openai            # real OpenAI (requires key)
    python main.py --provider anthropic         # real Anthropic (requires key)
    python main.py --calls 10                   # run more calls
    python main.py --show-calls                 # print every individual call
"""

import time
import random
from dataclasses import dataclass, field
from typing import Any

import click

from dd_llm import (
    LLMAdapter, LLMResponse,
    register_adapter, get_adapter, list_adapters,
)


# ---------------------------------------------------------------------------
# Configurable mock adapters with realistic cost profiles
# ---------------------------------------------------------------------------

# Approximate real-world rates ($ per 1 M tokens)
_COST_PROFILES: dict[str, dict[str, float]] = {
    "mock-fast":     {"input": 0.10,  "output": 0.30},   # e.g. GPT-4o-mini tier
    "mock-balanced": {"input": 2.50,  "output": 10.00},  # e.g. GPT-4o tier
    "mock-premium":  {"input": 15.00, "output": 75.00},  # e.g. Claude-3-Opus tier
}


class MockCostAdapter(LLMAdapter):
    """Simulates an LLM call with configurable latency, tokens, and cost."""

    def __init__(self, profile: str = "mock-balanced"):
        self.profile = profile
        self._rates = _COST_PROFILES.get(profile, _COST_PROFILES["mock-balanced"])

    def call(self, prompt="", *, model="", messages=None, system=None,
             max_tokens=4096, temperature=0.7, **kwargs):
        start = self._measure_time()

        history = messages or [{"role": "user", "content": prompt}]
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            prompt,
        )

        # Simulate variable token usage based on prompt length
        base_in = max(10, len(last_user.split()))
        input_tokens  = base_in + random.randint(5, 30)
        output_tokens = random.randint(20, 120)

        # Simulate latency: premium models are slower
        latency_base = {"mock-fast": 80, "mock-balanced": 300, "mock-premium": 800}
        latency_ms = latency_base.get(self.profile, 300) + random.uniform(-50, 100)
        time.sleep(latency_ms / 1000)  # actually pause so latency is realistic

        cost_usd = (
            input_tokens  * self._rates["input"]  / 1_000_000
            + output_tokens * self._rates["output"] / 1_000_000
        )

        reply = f"[{self.profile}] Response to: {last_user[:40]!r}"

        return LLMResponse(
            content=reply,
            success=True,
            provider=self.profile,
            model=self.profile,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=self._elapsed_ms(start),
            cost_usd=cost_usd,
        )


for _p in _COST_PROFILES:
    register_adapter(_p, lambda profile=_p: MockCostAdapter(profile))
# "mock" alias → balanced tier (default when no --provider given)
register_adapter("mock", lambda: MockCostAdapter("mock-balanced"))


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """Metadata for a single LLM call."""
    provider:      str
    model:         str
    prompt_snippet: str
    input_tokens:  int
    output_tokens: int
    latency_ms:    float
    cost_usd:      float
    success:       bool


@dataclass
class CostTracker:
    """Accumulate and report token usage and cost across LLM calls."""

    records: list[CallRecord] = field(default_factory=list)

    def record(self, result: LLMResponse, prompt_snippet: str = ""):
        self.records.append(CallRecord(
            provider=result.provider,
            model=result.model,
            prompt_snippet=prompt_snippet[:60],
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd or 0.0,
            success=result.success,
        ))

    # -- aggregate stats -------------------------------------------------------

    def total_calls(self) -> int:
        return len(self.records)

    def successful_calls(self) -> int:
        return sum(1 for r in self.records if r.success)

    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    def total_tokens(self) -> int:
        return self.total_input_tokens() + self.total_output_tokens()

    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def avg_latency_ms(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.latency_ms for r in self.records) / len(self.records)

    def by_provider(self) -> dict[str, dict[str, Any]]:
        """Per-provider aggregates."""
        agg: dict[str, dict[str, Any]] = {}
        for rec in self.records:
            if rec.provider not in agg:
                agg[rec.provider] = {
                    "calls": 0, "successes": 0,
                    "input_tokens": 0, "output_tokens": 0,
                    "cost_usd": 0.0, "total_latency_ms": 0.0,
                }
            a = agg[rec.provider]
            a["calls"]          += 1
            a["successes"]      += int(rec.success)
            a["input_tokens"]   += rec.input_tokens
            a["output_tokens"]  += rec.output_tokens
            a["cost_usd"]       += rec.cost_usd
            a["total_latency_ms"] += rec.latency_ms

        # Derive averages
        for name, a in agg.items():
            a["avg_latency_ms"] = a["total_latency_ms"] / max(a["calls"], 1)
            a["success_rate"]   = a["successes"] / max(a["calls"], 1)
            del a["total_latency_ms"], a["successes"]

        return agg

    def print_summary(self):
        """Print a formatted summary report."""
        width = 60
        print(f"\n{'═' * width}")
        print(f"  COST & USAGE SUMMARY")
        print(f"{'═' * width}")
        print(f"  Total calls       : {self.total_calls()}"
              f" ({self.successful_calls()} succeeded)")
        print(f"  Input tokens      : {self.total_input_tokens():,}")
        print(f"  Output tokens     : {self.total_output_tokens():,}")
        print(f"  Total tokens      : {self.total_tokens():,}")
        print(f"  Total cost        : ${self.total_cost_usd():.6f}")
        print(f"  Avg latency       : {self.avg_latency_ms():.0f} ms")

        if len(self.by_provider()) > 1:
            print(f"\n{'─' * width}")
            print(f"  BY PROVIDER")
            print(f"{'─' * width}")
            for prov, a in sorted(self.by_provider().items()):
                print(f"\n  [{prov}]")
                print(f"    calls        : {a['calls']}"
                      f"  (success rate {a['success_rate']:.0%})")
                print(f"    tokens       : {a['input_tokens']:,} in"
                      f" / {a['output_tokens']:,} out")
                print(f"    cost         : ${a['cost_usd']:.6f}")
                print(f"    avg latency  : {a['avg_latency_ms']:.0f} ms")

        print(f"{'═' * width}\n")

    def print_call_log(self):
        """Print every individual call record."""
        print(f"\n{'─' * 80}")
        print(f"  CALL LOG ({len(self.records)} records)")
        print(f"{'─' * 80}")
        print(f"  {'#':<4} {'provider':<18} {'in':>6} {'out':>6}"
              f" {'ms':>7} {'cost':>10}  prompt")
        print(f"  {'─'*4} {'─'*18} {'─'*6} {'─'*6} {'─'*7} {'─'*10}  {'─'*30}")
        for i, r in enumerate(self.records, 1):
            status = "" if r.success else " FAIL"
            print(f"  {i:<4} {r.provider:<18} {r.input_tokens:>6}"
                  f" {r.output_tokens:>6} {r.latency_ms:>7.0f}"
                  f" ${r.cost_usd:>9.6f}  {r.prompt_snippet!r}{status}")
        print()


# ---------------------------------------------------------------------------
# Demo prompts
# ---------------------------------------------------------------------------

_PROMPTS = [
    "What is machine learning?",
    "Explain gradient descent in one paragraph.",
    "What are the differences between SQL and NoSQL?",
    "Summarise the history of the internet.",
    "How does a transformer model work?",
    "What is retrieval-augmented generation?",
    "Describe the CAP theorem.",
    "What is a vector database used for?",
    "Explain the difference between precision and recall.",
    "What are embeddings in the context of NLP?",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--provider",    default="mock",    help="Primary LLM provider")
@click.option("--model",       default="",        help="Model override")
@click.option("--calls",       default=6,         show_default=True,
              help="Number of demo calls to make")
@click.option("--multi",       is_flag=True,
              help="Spread calls across all three mock-* cost profiles")
@click.option("--show-calls",  is_flag=True,
              help="Print per-call log after the summary")
def main(provider, model, calls, multi, show_calls):
    """Cost tracking: aggregate token usage and cost across LLM calls."""

    print(f"Available providers: {list_adapters()}")

    tracker = CostTracker()

    if multi:
        # Round-robin across fast / balanced / premium mocks
        profiles = ["mock-fast", "mock-balanced", "mock-premium"]
        print(f"Multi-provider mode: {profiles}\n")
        for i in range(calls):
            prov   = profiles[i % len(profiles)]
            prompt = _PROMPTS[i % len(_PROMPTS)]
            print(f"  Call {i+1}/{calls}  [{prov}]  {prompt[:50]!r}")
            adapter = get_adapter(prov)
            result  = adapter.call(prompt=prompt, model=model)
            tracker.record(result, prompt_snippet=prompt)
    else:
        print(f"Using provider: {provider}\n")
        adapter = get_adapter(provider)
        for i in range(calls):
            prompt = _PROMPTS[i % len(_PROMPTS)]
            print(f"  Call {i+1}/{calls}  {prompt[:50]!r}")
            result = adapter.call(prompt=prompt, model=model)
            tracker.record(result, prompt_snippet=prompt)

    if show_calls:
        tracker.print_call_log()

    tracker.print_summary()


if __name__ == "__main__":
    main()
