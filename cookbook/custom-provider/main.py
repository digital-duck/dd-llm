"""dd-llm Custom Provider — register your own LLM adapter.

Demonstrates dd-llm's extensibility: define a custom adapter class,
register it, and use it through call_llm() and UnifiedLLMProvider
just like any built-in provider.

Usage:
    python main.py
    python main.py --provider echo         # Use the custom echo provider
    python main.py --provider uppercase    # Use the custom uppercase provider
"""

import time
import click
from dd_llm import (
    LLMAdapter, LLMResponse,
    register_adapter, call_llm, list_adapters, UnifiedLLMProvider,
)


# ---------------------------------------------------------------------------
# Custom adapter 1: Echo (returns the prompt back)
# ---------------------------------------------------------------------------

class EchoAdapter(LLMAdapter):
    """A trivial adapter that echoes back the prompt. Useful for testing."""

    def call(self, prompt="", *, model="", messages=None, **kwargs):
        start = self._measure_time()

        # Extract text from messages or prompt
        if messages:
            text = " | ".join(m["content"] for m in messages)
        else:
            text = prompt

        return LLMResponse(
            content=f"[echo] {text}",
            success=True,
            provider="echo",
            model="echo-v1",
            input_tokens=len(text.split()),
            output_tokens=len(text.split()),
            latency_ms=self._elapsed_ms(start),
            cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# Custom adapter 2: Uppercase (transforms prompt to uppercase)
# ---------------------------------------------------------------------------

class UppercaseAdapter(LLMAdapter):
    """A mock adapter that uppercases the input. Demonstrates transformation."""

    def __init__(self, delay_ms: float = 0):
        self.delay_ms = delay_ms

    def call(self, prompt="", *, model="", messages=None, **kwargs):
        start = self._measure_time()

        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000)

        if messages:
            text = messages[-1]["content"]
        else:
            text = prompt

        return LLMResponse(
            content=text.upper(),
            success=True,
            provider="uppercase",
            model="upper-v1",
            input_tokens=len(text.split()),
            output_tokens=len(text.split()),
            latency_ms=self._elapsed_ms(start),
            cost_usd=0.0,
        )

    def list_models(self):
        return ["upper-v1"]


# ---------------------------------------------------------------------------
# Register both adapters
# ---------------------------------------------------------------------------

register_adapter("echo", EchoAdapter)
register_adapter("uppercase", UppercaseAdapter)


@click.command()
@click.option("--provider", default="echo", help="Provider to use")
@click.argument("prompt", default="Hello from a custom dd-llm provider!")
def main(provider, prompt):
    """Demonstrate custom LLM adapters registered with dd-llm."""

    print(f"All providers: {list_adapters()}")
    print()

    # --- Use via call_llm ---
    print(f"=== call_llm(provider='{provider}') ===")
    try:
        response = call_llm(prompt, provider=provider)
        print(f"Response: {response}")
    except RuntimeError as e:
        print(f"Error: {e}")

    print()

    # --- Use via UnifiedLLMProvider with custom fallback chain ---
    print("=== UnifiedLLMProvider with custom providers ===")
    unified = UnifiedLLMProvider(
        primary_provider="echo",
        fallback_providers=["uppercase"],
        max_retries=1,
    )
    result = unified.call(prompt)
    print(f"Provider: {result.provider}")
    print(f"Content:  {result.content}")
    print(f"Latency:  {result.latency_ms:.2f}ms")

    print()

    # --- Show that custom providers appear in stats ---
    unified.call("Another call")
    print("Stats:")
    for name, stats in unified.get_provider_stats().items():
        print(f"  {name}: {stats['successes']} successes")


if __name__ == "__main__":
    main()
