"""
EpisodeReloader — automated game reset for ONI-AI training runs.

Sequence per reset:
  1. Write autoload.txt on game host (SSH)
  2. Graceful quit via xdotool Alt+F4
  3. Wait for process death
  4. Launch game via Steam URL
  5. Poll TCP :9999 until ONIBridge is live

Usage:
    reloader = EpisodeReloader()
    result = await reloader.reset_episode()
    if not result.success:
        raise RuntimeError(result.error)
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional

import asyncssh

GAME_HOST = "10.0.0.10"
GAME_USER = "myroproductions"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
DISPLAY = ":0"
GAME_PROCESS = "OxygenNotIncluded"
BRIDGE_PORT = 9999

SAVE_FILES_DIR = (
    "/home/myroproductions/.config/unity3d/Klei/"
    "Oxygen Not Included/save_files"
)
AUTOLOAD_CONFIG = (
    "/home/myroproductions/.config/unity3d/Klei/"
    "Oxygen Not Included/mods/Dev/ONIBridge/autoload.txt"
)

# Canonical training save — colony must be named exactly "training-start"
# so the path is deterministic: save_files/training-start/training-start.sav
CANONICAL_COLONY = "training-start"
CANONICAL_SAVE = f"{SAVE_FILES_DIR}/{CANONICAL_COLONY}/{CANONICAL_COLONY}.sav"


@dataclass
class ReloadResult:
    success: bool
    elapsed_s: float
    error: Optional[str] = None


class EpisodeReloader:
    """Handles full episode reset: quit game -> restart -> autoload save -> bridge ready."""

    def __init__(self, save_path: str = CANONICAL_SAVE):
        self.save_path = save_path

    async def reset_episode(self) -> ReloadResult:
        """Run the full reset sequence. Returns ReloadResult with success/elapsed/error."""
        start = time.monotonic()
        try:
            async with asyncssh.connect(
                GAME_HOST,
                username=GAME_USER,
                client_keys=[SSH_KEY],
                known_hosts=None,
            ) as conn:
                await self._write_autoload(conn)

                quit_ok = await self._quit_game(conn)
                if not quit_ok:
                    await self._kill_game(conn)

                await self._wait_for_process_death(conn)
                await self._launch_game(conn)

            # Bridge polling is outside the SSH context — game host connection not needed
            await self._wait_for_bridge_ready()

            elapsed = time.monotonic() - start
            return ReloadResult(success=True, elapsed_s=elapsed)

        except Exception as e:
            elapsed = time.monotonic() - start
            return ReloadResult(success=False, elapsed_s=elapsed, error=str(e))

    async def _write_autoload(self, conn: asyncssh.SSHClientConnection) -> None:
        """Write the save path to autoload.txt on the game host via SFTP."""
        async with conn.start_sftp_client() as sftp:
            async with await sftp.open(AUTOLOAD_CONFIG, "w") as f:
                await f.write(self.save_path)

    async def _quit_game(self, conn: asyncssh.SSHClientConnection) -> bool:
        """Attempt graceful quit via xdotool Alt+F4. Returns True if window was found."""
        result = await conn.run(
            f"DISPLAY={DISPLAY} xdotool search --name 'Oxygen Not Included'",
            check=False,
        )
        if result.exit_status != 0 or not result.stdout.strip():
            return False

        window_id = result.stdout.strip().splitlines()[-1]
        await conn.run(
            f"DISPLAY={DISPLAY} xdotool key --window {window_id} alt+F4",
            check=False,
        )
        await asyncio.sleep(5)
        return True

    async def _kill_game(self, conn: asyncssh.SSHClientConnection) -> None:
        """Force-kill the game process."""
        await conn.run(f"pkill -SIGTERM -f {GAME_PROCESS}", check=False)
        await asyncio.sleep(3)
        await conn.run(f"pkill -SIGKILL -f {GAME_PROCESS}", check=False)

    async def _wait_for_process_death(
        self, conn: asyncssh.SSHClientConnection, timeout_s: int = 15
    ) -> None:
        """Poll until the game process is gone. Force-kills after timeout."""
        for _ in range(timeout_s):
            result = await conn.run(f"pgrep -f {GAME_PROCESS}", check=False)
            if result.exit_status != 0:
                return
            await asyncio.sleep(1)
        await self._kill_game(conn)

    async def _launch_game(self, conn: asyncssh.SSHClientConnection) -> None:
        """Launch ONI via Steam URL. Fire-and-forget."""
        await conn.run(
            f"DISPLAY={DISPLAY} nohup steam steam://rungameid/457140 </dev/null >/dev/null 2>&1 &",
            check=False,
        )
        await asyncio.sleep(5)

    async def _wait_for_bridge_ready(
        self, timeout_s: int = 120, poll_interval_s: float = 2.0
    ) -> None:
        """
        Poll TCP :9999 on game host until ONIBridge accepts connections and
        sends a valid state message. This is the definitive 'game is ready' signal.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            writer = None
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(GAME_HOST, BRIDGE_PORT),
                    timeout=3.0,
                )
                line = await asyncio.wait_for(reader.readline(), timeout=10.0)
                if b'"type"' in line and b'"state"' in line:
                    return
            except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
                pass
            finally:
                if writer is not None:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except Exception:
                        pass
            await asyncio.sleep(poll_interval_s)
        raise TimeoutError(
            f"ONIBridge did not come up within {timeout_s}s after game launch"
        )
