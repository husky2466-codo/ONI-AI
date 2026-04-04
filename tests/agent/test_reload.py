"""Tests for EpisodeReloader — unit tests only, no real SSH/game connections."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.reload import EpisodeReloader, CANONICAL_SAVE, ReloadResult


class TestReloadResult:
    def test_success_fields(self):
        r = ReloadResult(success=True, elapsed_s=42.3)
        assert r.success is True
        assert r.elapsed_s == pytest.approx(42.3)
        assert r.error is None

    def test_failure_fields(self):
        r = ReloadResult(success=False, elapsed_s=5.0, error="timeout")
        assert r.success is False
        assert r.error == "timeout"


class TestEpisodeReloaderInit:
    def test_default_save_path(self):
        r = EpisodeReloader()
        assert r.save_path == CANONICAL_SAVE
        assert "training-start" in r.save_path

    def test_custom_save_path(self):
        r = EpisodeReloader(save_path="/custom/path/save.sav")
        assert r.save_path == "/custom/path/save.sav"


class TestEpisodeReloaderReset:
    @pytest.mark.asyncio
    async def test_reset_returns_failure_on_ssh_error(self):
        """SSH connection failure -> ReloadResult(success=False)."""
        reloader = EpisodeReloader()
        with patch("src.agent.reload.asyncssh.connect", side_effect=OSError("refused")):
            result = await reloader.reset_episode()
        assert result.success is False
        assert "refused" in result.error
        assert result.elapsed_s >= 0

    @pytest.mark.asyncio
    async def test_wait_for_bridge_ready_timeout(self):
        """Bridge poll timeout raises TimeoutError (caught by reset_episode)."""
        reloader = EpisodeReloader()
        with patch(
            "src.agent.reload.asyncio.open_connection",
            side_effect=ConnectionRefusedError,
        ):
            with pytest.raises(TimeoutError):
                await reloader._wait_for_bridge_ready(timeout_s=1, poll_interval_s=0.1)

    @pytest.mark.asyncio
    async def test_wait_for_bridge_ready_succeeds_on_valid_state(self):
        """Bridge poll succeeds when it receives a line containing state JSON."""
        reloader = EpisodeReloader()

        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(
            return_value=b'{"type": "state", "data": {"cycle": 1}}\n'
        )
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "src.agent.reload.asyncio.open_connection",
            return_value=(mock_reader, mock_writer),
        ):
            # Should complete without raising
            await reloader._wait_for_bridge_ready(timeout_s=5, poll_interval_s=0.1)

    @pytest.mark.asyncio
    async def test_write_autoload_uses_sftp_not_shell(self):
        """_write_autoload should write via SFTP, not a shell command."""
        reloader = EpisodeReloader(save_path="/path/with 'quotes' and spaces/save.sav")

        mock_sftp = AsyncMock()
        mock_file = AsyncMock()
        mock_sftp.open = AsyncMock(return_value=mock_file)
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=False)
        mock_file.write = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.start_sftp_client = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_sftp),
            __aexit__=AsyncMock(return_value=False),
        ))

        await reloader._write_autoload(mock_conn)

        mock_sftp.open.assert_called_once_with(
            "/home/myroproductions/.config/unity3d/Klei/"
            "Oxygen Not Included/mods/Dev/ONIBridge/autoload.txt",
            "w",
        )
        mock_file.write.assert_called_once_with("/path/with 'quotes' and spaces/save.sav")
