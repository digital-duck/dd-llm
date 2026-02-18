"""Claude Code CLI adapter: wraps the ``claude`` CLI for development use.

Leverages Claude Code subscription billing for zero marginal cost during
development.  Invokes ``claude -p "<prompt>"`` via subprocess.
"""

from __future__ import annotations

import subprocess

from dd_llm.base import LLMAdapter, LLMResponse


class ClaudeCLIAdapter(LLMAdapter):
    """LLM adapter that wraps the Claude Code CLI.

    Designed for development use — leverages existing Claude Code subscription
    (flat billing = zero marginal cost per call).
    """

    def __init__(
        self,
        cli_path: str = "claude",
        timeout: int = 300,
        allowed_tools: list[str] | None = None,
    ):
        self.cli_path = cli_path
        self.timeout = timeout
        self.allowed_tools = allowed_tools or []

    def call(
        self,
        prompt: str = "",
        *,
        model: str = "",
        messages: list[dict] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Generate response by invoking the claude CLI."""
        start = self._measure_time()

        # Build effective prompt from messages or prompt string
        if messages:
            full_prompt = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )
        else:
            full_prompt = prompt

        if system:
            full_prompt = f"System: {system}\n\nUser: {full_prompt}"

        # Build CLI command
        cmd = [self.cli_path, "-p", full_prompt]
        if self.allowed_tools:
            cmd += ["--allowedTools", ",".join(self.allowed_tools)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            return LLMResponse(
                content="",
                success=False,
                provider="claude_cli",
                model="claude-cli",
                error_history=[{
                    "error": f"Claude CLI not found at '{self.cli_path}'. "
                             "Install Claude Code: https://docs.anthropic.com/en/docs/claude-code",
                    "error_type": "FileNotFoundError",
                }],
            )
        except subprocess.TimeoutExpired:
            return LLMResponse(
                content="",
                success=False,
                provider="claude_cli",
                model="claude-cli",
                error_history=[{
                    "error": f"Claude CLI timed out after {self.timeout}s",
                    "error_type": "TimeoutError",
                }],
            )

        latency = self._elapsed_ms(start)

        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors="replace").strip()
            return LLMResponse(
                content="",
                success=False,
                provider="claude_cli",
                model="claude-cli",
                latency_ms=latency,
                error_history=[{
                    "error": f"Claude CLI error (exit {result.returncode}): {error_msg}",
                    "error_type": "RuntimeError",
                }],
            )

        content = result.stdout.decode("utf-8", errors="replace").strip()

        # Estimate tokens (~4 chars per token)
        input_tokens = max(1, len(full_prompt) // 4)
        output_tokens = max(1, len(content) // 4) if content else 0

        return LLMResponse(
            content=content,
            success=True,
            provider="claude_cli",
            model="claude-cli",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            cost_usd=0.0,
        )

    def list_models(self) -> list[str]:
        return ["claude-cli"]
