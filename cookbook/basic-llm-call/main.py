"""dd-llm Basic LLM Call — call any LLM with a single function.

Usage:
    python main.py                                              # default provider (from LLM_PROVIDER env)
    python main.py --provider claude_cli                        # Claude CLI (free via subscription)
    python main.py --provider openai --model gpt-4o             # OpenAI
    python main.py --provider anthropic                         # Anthropic
    python main.py --provider ollama --model llama3.2           # Local Ollama
    python main.py --provider gemini                            # Google Gemini
    python main.py "Explain quantum computing in one sentence"  # Custom prompt
"""

import click
from dd_llm import call_llm, get_adapter, list_adapters, UnifiedLLMProvider


@click.command()
@click.argument("prompt", default="What is the meaning of life? Answer in one sentence.")
@click.option("--provider", default=None, help="LLM provider name")
@click.option("--model", default=None, help="Model name (provider-specific)")
def main(prompt, provider, model):
    """Call any LLM provider with a single function."""

    print(f"Available providers: {list_adapters()}")
    print()

    # --- Method 1: Simple call_llm convenience function ---
    print("=== Method 1: call_llm() ===")
    try:
        kwargs = {}
        if provider:
            kwargs["provider"] = provider
        if model:
            kwargs["model"] = model
        response = call_llm(prompt, **kwargs)
        print(f"Response: {response[:200]}")
    except RuntimeError as e:
        print(f"Error: {e}")

    print()

    # --- Method 2: Direct adapter usage (more control) ---
    print("=== Method 2: Direct adapter ===")
    adapter_kwargs = {}
    if model:
        adapter_kwargs["default_model"] = model
    adapter = get_adapter(provider or "openai", **adapter_kwargs)
    result = adapter.call(prompt)

    if result.success:
        print(f"Provider: {result.provider}")
        print(f"Model:    {result.model}")
        print(f"Tokens:   {result.input_tokens} in, {result.output_tokens} out")
        print(f"Latency:  {result.latency_ms:.0f}ms")
        print(f"Response: {result.content[:200]}")
    else:
        print(f"Failed: {result.error_history}")

    print()

    # --- Method 3: Using messages (conversation format) ---
    print("=== Method 3: Messages format ===")
    try:
        response = call_llm(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be concise."},
                {"role": "user", "content": prompt},
            ],
            provider=provider,
            model=model,
        )
        print(f"Response: {response[:200]}")
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
