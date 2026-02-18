"""Tests for ClaudeCLIAdapter (mocked subprocess)."""

from unittest.mock import patch, MagicMock
import subprocess

import pytest

from dd_llm.adapters.claude_cli import ClaudeCLIAdapter


class TestClaudeCLIAdapter:
    def test_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"Hello from Claude!"
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            adapter = ClaudeCLIAdapter()
            resp = adapter.call("What is 2+2?")

            assert resp.success
            assert resp.content == "Hello from Claude!"
            assert resp.provider == "claude_cli"
            assert resp.cost_usd == 0.0
            mock_run.assert_called_once()

    def test_cli_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            adapter = ClaudeCLIAdapter()
            resp = adapter.call("hello")

            assert not resp.success
            assert "not found" in resp.error_history[0]["error"]

    def test_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 300)):
            adapter = ClaudeCLIAdapter()
            resp = adapter.call("hello")

            assert not resp.success
            assert "timed out" in resp.error_history[0]["error"]

    def test_nonzero_exit(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b""
        mock_result.stderr = b"Some error"

        with patch("subprocess.run", return_value=mock_result):
            adapter = ClaudeCLIAdapter()
            resp = adapter.call("hello")

            assert not resp.success
            assert "exit 1" in resp.error_history[0]["error"]

    def test_system_prompt(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"response"
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            adapter = ClaudeCLIAdapter()
            adapter.call("hello", system="Be helpful")

            # Verify system prompt is included in the CLI call
            cmd = mock_run.call_args[0][0]
            prompt_arg = cmd[2]  # -p argument
            assert "System: Be helpful" in prompt_arg

    def test_messages_param(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"response"
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            adapter = ClaudeCLIAdapter()
            adapter.call(messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "bye"},
            ])

            cmd = mock_run.call_args[0][0]
            prompt_arg = cmd[2]
            assert "user: hi" in prompt_arg
            assert "assistant: hello" in prompt_arg

    def test_allowed_tools(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"ok"
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            adapter = ClaudeCLIAdapter(allowed_tools=["Read", "Write"])
            adapter.call("hello")

            cmd = mock_run.call_args[0][0]
            assert "--allowedTools" in cmd
            assert "Read,Write" in cmd

    def test_list_models(self):
        adapter = ClaudeCLIAdapter()
        assert adapter.list_models() == ["claude-cli"]
