# tests/agent/test_ui_actions.py
"""Tests for UI actions (xdotool via SSH) in the runner."""
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from src.agent.runner import open_settings_via_xdotool


def test_open_settings_runs_ssh_xdotool():
    """open_settings_via_xdotool should SSH to the game host and click the cog."""
    async def _run():
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            proc = MagicMock()
            proc.wait = AsyncMock(return_value=0)
            mock_exec.return_value = proc

            result = await open_settings_via_xdotool("10.0.0.10")

            assert result is True
            mock_exec.assert_called_once()
            cmd = mock_exec.call_args[0]
            assert "ssh" in cmd
            assert "myroproductions@10.0.0.10" in cmd
            full_cmd = " ".join(str(a) for a in cmd)
            assert "xdotool" in full_cmd
            assert "1893" in full_cmd
            assert "88" in full_cmd

    asyncio.run(_run())


def test_open_settings_returns_false_on_ssh_failure():
    """open_settings_via_xdotool should return False when SSH exits non-zero."""
    async def _run():
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            proc = MagicMock()
            proc.wait = AsyncMock(return_value=1)
            mock_exec.return_value = proc

            result = await open_settings_via_xdotool("10.0.0.10")
            assert result is False

    asyncio.run(_run())


def test_open_settings_returns_false_on_exception():
    """open_settings_via_xdotool should return False when SSH raises."""
    async def _run():
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("ssh not found")):
            result = await open_settings_via_xdotool("10.0.0.10")
            assert result is False

    asyncio.run(_run())
