"""dd-llm Multi-Turn Chat — maintain conversation history across LLM calls.

Demonstrates:
- Building a message history list manually
- Passing the full history on every call via the messages= parameter
- Adding system prompt to set persona / tone
- Inspecting per-turn token usage from LLMResponse
- Running without a real API key using the built-in mock EchoAdapter

Usage:
    python main.py                              # demo mode (mock adapter, no API key)
    python main.py --provider openai            # real OpenAI
    python main.py --provider anthropic         # real Anthropic
    python main.py --provider ollama            # local Ollama
    python main.py --interactive --provider openai   # type your own messages
"""

import time
import click

from dd_llm import (
    LLMAdapter, LLMResponse,
    register_adapter, get_adapter, list_adapters, UnifiedLLMProvider,
)


# ---------------------------------------------------------------------------
# Mock adapter — works without any API key, useful for demo / testing
# ---------------------------------------------------------------------------

class MockChatAdapter(LLMAdapter):
    """Returns a canned reply that references the last user message.
    Useful for demo / CI without a real API key.
    """

    def call(self, prompt="", *, model="", messages=None, system=None,
             max_tokens=4096, temperature=0.7, **kwargs):
        start = self._measure_time()
        history = messages or [{"role": "user", "content": prompt}]
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            prompt,
        )
        turn = sum(1 for m in history if m["role"] == "assistant") + 1
        reply = (
            f"[mock turn {turn}] You said: '{last_user[:60]}'. "
            f"History has {len(history)} messages."
        )
        return LLMResponse(
            content=reply,
            success=True,
            provider="mock",
            model="mock-chat-v1",
            input_tokens=sum(len(m["content"].split()) for m in history),
            output_tokens=len(reply.split()),
            latency_ms=self._elapsed_ms(start),
            cost_usd=0.0,
        )


register_adapter("mock", MockChatAdapter)


# ---------------------------------------------------------------------------
# Chat session helper
# ---------------------------------------------------------------------------

class ChatSession:
    """Maintains conversation history and calls the LLM on each turn."""

    def __init__(self, provider: str, system: str = "", model: str = ""):
        self.adapter = get_adapter(provider)
        self.model = model
        self.history: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        if system:
            self.history.append({"role": "system", "content": system})

    def chat(self, user_message: str) -> LLMResponse:
        """Add user message to history and get LLM reply."""
        self.history.append({"role": "user", "content": user_message})

        result = self.adapter.call(
            messages=self.history,
            model=self.model,
        )

        if result.success:
            self.history.append({"role": "assistant", "content": result.content})
            self.total_input_tokens += result.input_tokens
            self.total_output_tokens += result.output_tokens
            self.total_cost += result.cost_usd or 0.0

        return result

    def summary(self) -> dict:
        return {
            "turns": sum(1 for m in self.history if m["role"] == "user"),
            "total_messages": len(self.history),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--provider", default="mock", help="LLM provider name")
@click.option("--model", default="", help="Model override")
@click.option("--interactive", is_flag=True, help="Enter interactive chat mode")
def main(provider, model, interactive):
    """Multi-turn chat with persistent message history."""

    print(f"Available providers: {list_adapters()}")
    print(f"Using provider: {provider}")
    print()

    system_prompt = (
        "You are a knowledgeable but concise assistant. "
        "Answer in 1-3 sentences. Be direct."
    )

    # ------------------------------------------------------------------
    # Demo mode: pre-scripted conversation
    # ------------------------------------------------------------------
    if not interactive:
        print("=== Demo: pre-scripted 3-turn conversation ===")
        print(f"System: {system_prompt}")
        print()

        session = ChatSession(provider=provider, system=system_prompt, model=model)

        turns = [
            "What is a large language model?",
            "How does it differ from a traditional search engine?",
            "Can you give me one concrete use-case where LLMs beat search?",
        ]

        for user_msg in turns:
            print(f"User:      {user_msg}")
            result = session.chat(user_msg)
            if result.success:
                print(f"Assistant: {result.content}")
                print(f"           [{result.input_tokens} in / {result.output_tokens} out"
                      f" — {result.latency_ms:.0f}ms]")
            else:
                print(f"ERROR: {(result.error_history or [{}])[-1].get('error')}")
            print()

        print("=== Session summary ===")
        for k, v in session.summary().items():
            print(f"  {k}: {v}")

        print()
        print("=== Full message history ===")
        for i, msg in enumerate(session.history):
            role = msg["role"].upper()
            snippet = msg["content"][:80].replace("\n", " ")
            print(f"  [{i}] {role:<12} {snippet}")

        return

    # ------------------------------------------------------------------
    # Interactive mode: read from stdin
    # ------------------------------------------------------------------
    print(f"=== Interactive chat (provider={provider}) ===")
    print(f"System: {system_prompt}")
    print("Type your message and press Enter. Ctrl-C or 'quit' to exit.\n")

    session = ChatSession(provider=provider, system=system_prompt, model=model)

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_msg or user_msg.lower() in ("quit", "exit", "q"):
            break

        result = session.chat(user_msg)
        if result.success:
            print(f"Assistant: {result.content}")
            print(f"[{result.input_tokens} in / {result.output_tokens} out"
                  f" — {result.latency_ms:.0f}ms]")
        else:
            last_err = (result.error_history or [{}])[-1].get("error", "unknown")
            print(f"[ERROR] {last_err}")
        print()

    print("\n=== Session summary ===")
    for k, v in session.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
