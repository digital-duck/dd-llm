"""dd-llm Provider Fallback — retry with exponential backoff + automatic failover.

Demonstrates UnifiedLLMProvider's self-healing behavior:
1. Try the primary provider with retries (exponential backoff + jitter)
2. On exhaustion, fall back to the next provider in the chain
3. On retries, inject error context so the LLM can self-correct

Usage:
    python main.py
    python main.py --primary ollama --fallback openai --fallback anthropic
    python main.py --retries 2 --initial-wait 0.5
"""

import click
from dd_llm import UnifiedLLMProvider


@click.command()
@click.option("--primary", default="openai", help="Primary provider")
@click.option("--fallback", multiple=True, default=["anthropic", "ollama"], help="Fallback providers (repeatable)")
@click.option("--retries", default=3, help="Max retries per provider")
@click.option("--initial-wait", default=1.0, help="Initial backoff seconds")
def main(primary, fallback, retries, initial_wait):
    """Demonstrate retry + fallback across multiple LLM providers."""

    provider = UnifiedLLMProvider(
        primary_provider=primary,
        fallback_providers=list(fallback),
        max_retries=retries,
        initial_wait=initial_wait,
    )

    prompt = "What are the three laws of thermodynamics? One sentence each."

    print(f"Primary:   {primary}")
    print(f"Fallback:  {list(fallback)}")
    print(f"Retries:   {retries} per provider")
    print(f"Backoff:   {initial_wait}s initial")
    print()

    # Make the call — retries and fallback happen automatically
    result = provider.call(prompt)

    if result.success:
        print(f"Provider:  {result.provider}")
        print(f"Model:     {result.model}")
        print(f"Attempts:  {result.attempts}")
        print(f"Time:      {result.total_time:.2f}s")
        print()
        print(result.content)
    else:
        print(f"All providers failed after {result.attempts} total attempts.")
        print(f"Time: {result.total_time:.2f}s")
        print()
        print("Error history:")
        for err in result.error_history or []:
            print(f"  [{err['provider']}] attempt {err['attempt']}: "
                  f"{err['error_type']}: {err['error'][:100]}")

    # Show stats
    print()
    print("Provider stats:")
    for name, stats in provider.get_provider_stats().items():
        print(f"  {name}: {stats['successes']} ok, {stats['failures']} fail, "
              f"rate={stats['success_rate']:.0%}, avg={stats['avg_time']:.2f}s")


if __name__ == "__main__":
    main()
