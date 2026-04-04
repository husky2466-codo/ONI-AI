# Game Reload Automation — Design Spec

**Date:** 2026-04-04
**Status:** Approved for implementation
**Priority:** P1 — implement immediately after episode logging is confirmed working

---

## Overview

Manual game reloads cap training throughput at 2–4 episodes per hour. Automated reloads
target 40–80 episodes per hour (45–90 second reset cycle), enabling real RL training runs
overnight without human intervention.

This spec defines the full episode reset sequence: graceful game shutdown, restart via
Steam, mod-side save autoload, and Python-side readiness detection.

---

## Verified Infrastructure

From direct inspection of the Linux game host (`myroproductions@10.0.0.10`):

| Item | Path / Value |
|------|-------------|
| Save files | `~/.config/unity3d/Klei/Oxygen Not Included/save_files/` |
| Mod DLL | `~/.config/unity3d/Klei/Oxygen Not Included/mods/Dev/ONIBridge/ONIBridge.dll` |
| Launch command | `steam steam://rungameid/457140` |
| Display | `:0` (physical X11 display, no headless mode) |
| SSH access | `ssh myroproductions@10.0.0.10` (key: `~/.ssh/id_ed25519`) |

**Why Steam URL and not direct binary:** The desktop shortcut launches via
`steam steam://rungameid/457140`. Using this same mechanism for automation ensures
identical launch conditions — same Steam overlay, same mod loading path, same environment.
Direct binary launch risks missing Steam initialization steps that mods may depend on.

---

## Architecture

```
Mac Mini (runner.py)                    Linux Desktop (game host)
─────────────────────────               ─────────────────────────
EpisodeReloader                         ONIBridge mod
  detects episode end               →     reads autoload.txt on first tick
  writes autoload.txt (SSH)         →     calls SaveLoader.Instance.Load()
  sends graceful quit (SSH xdotool) →     game closes
  waits for process death           ←     process exits
  launches game (SSH steam URL)     →     Steam launches ONI
  polls TCP :9999                   ←     mod opens TCP server
  receives first state message      ←     world loaded, episode begins
```

---

## Part 1: Mod-Side Autoload (C# — ONIBridge)

### New File: `AutoloadConfig.cs`

The mod reads a config file on the first game tick where the world is fully loaded.
If the file exists and contains a valid save path, it loads that save and deletes the
config file (one-shot — prevents re-loading on every restart if the file is left behind).

```csharp
using System.IO;
using UnityEngine;

namespace ONIBridge
{
    public static class AutoloadConfig
    {
        // Sits next to the mod DLL
        private static readonly string ConfigPath = Path.Combine(
            Application.persistentDataPath,
            "mods", "Dev", "ONIBridge", "autoload.txt"
        );

        // Called from GameTickPatch once world is loaded (Game.Instance != null
        // and SaveLoader.Instance != null and GameClock.Instance != null)
        public static void TryAutoload()
        {
            if (!File.Exists(ConfigPath)) return;

            string savePath = File.ReadAllText(ConfigPath).Trim();
            File.Delete(ConfigPath);  // consume immediately — one-shot

            if (string.IsNullOrEmpty(savePath)) return;
            if (!File.Exists(savePath))
            {
                Debug.LogWarning($"[ONIBridge] autoload.txt pointed to missing save: {savePath}");
                return;
            }

            Debug.Log($"[ONIBridge] Autoloading save: {savePath}");
            // SaveLoader.Instance.Load() is the standard ONI modding API for loading saves.
            // Verify exact signature via decompile — likely Load(string filename) or
            // Load(string filename, bool setSaveDir).
            SaveLoader.Instance.Load(savePath);
        }
    }
}
```

### Changes to `GameTickPatch.cs`

Call `AutoloadConfig.TryAutoload()` in the Harmony postfix, but only once and only after
the world is ready. Guard to prevent calling every tick:

```csharp
public static class GameTickPatch
{
    private static bool _autoloadAttempted = false;

    [HarmonyPostfix]
    static void Postfix()
    {
        // Existing action drain + state send logic...
        BridgeServer.DrainActions();

        // Autoload: attempt once after world is confirmed loaded
        if (!_autoloadAttempted
            && Game.Instance != null
            && SaveLoader.Instance != null
            && GameClock.Instance != null)
        {
            _autoloadAttempted = true;
            AutoloadConfig.TryAutoload();
        }

        // State serialization (existing)
        // ...
    }
}
```

**Reset `_autoloadAttempted` on mod unload** so it fires correctly on the next game
restart. Add to mod teardown / `OnDisable` if applicable.

---

## Part 2: Python — `EpisodeReloader`

### New File: `src/agent/reload.py`

```python
import asyncio
import asyncssh
import os
import time
from dataclasses import dataclass
from typing import Optional

GAME_HOST        = "10.0.0.10"
GAME_USER        = "myroproductions"
SSH_KEY          = os.path.expanduser("~/.ssh/id_ed25519")
DISPLAY          = ":0"
STEAM_LAUNCH_CMD = f"DISPLAY={DISPLAY} steam steam://rungameid/457140"
GAME_PROCESS     = "OxygenNotIncluded"
BRIDGE_PORT      = 9999

SAVE_FILES_DIR   = (
    "/home/myroproductions/.config/unity3d/Klei/"
    "Oxygen Not Included/save_files"
)
AUTOLOAD_CONFIG  = (
    "/home/myroproductions/.config/unity3d/Klei/"
    "Oxygen Not Included/mods/Dev/ONIBridge/autoload.txt"
)

# Save file format (Spaced Out DLC):
#   save_files/{colony-name}/{colony-name}.sav   ← main save
#   save_files/{colony-name}/{colony-name}.png   ← preview screenshot
#   save_files/{colony-name}/auto_save/          ← auto-saves
#
# SaveLoader.Instance.Load() takes the path to the .sav file directly.
# Canonical training save — update colony name after creating the training save:
CANONICAL_COLONY = "training-start"
CANONICAL_SAVE   = f"{SAVE_FILES_DIR}/{CANONICAL_COLONY}/{CANONICAL_COLONY}.sav"


@dataclass
class ReloadResult:
    success: bool
    elapsed_s: float
    error: Optional[str] = None


class EpisodeReloader:
    """Handles full episode reset: quit → restart → autoload → ready."""

    def __init__(self, save_path: str = CANONICAL_SAVE):
        self.save_path = save_path

    async def reset_episode(self) -> ReloadResult:
        start = time.monotonic()
        try:
            async with asyncssh.connect(
                GAME_HOST, username=GAME_USER,
                client_keys=[SSH_KEY], known_hosts=None
            ) as conn:
                # Step 1: Write autoload config before killing game
                await self._write_autoload(conn)

                # Step 2: Graceful quit
                quit_ok = await self._quit_game(conn)
                if not quit_ok:
                    await self._kill_game(conn)  # force kill fallback

                # Step 3: Wait for process to die
                await self._wait_for_process_death(conn)

                # Step 4: Launch game via Steam
                await self._launch_game(conn)

            # Step 5: Wait for TCP bridge to come up (outside SSH context)
            await self._wait_for_bridge_ready()

            elapsed = time.monotonic() - start
            return ReloadResult(success=True, elapsed_s=elapsed)

        except Exception as e:
            elapsed = time.monotonic() - start
            return ReloadResult(success=False, elapsed_s=elapsed, error=str(e))

    async def _write_autoload(self, conn) -> None:
        """Write save path to autoload.txt on the game host."""
        escaped = self.save_path.replace("'", "'\"'\"'")
        await conn.run(f"echo '{escaped}' > '{AUTOLOAD_CONFIG}'")

    async def _quit_game(self, conn) -> bool:
        """Attempt graceful quit via xdotool Alt+F4 on the game window."""
        try:
            # Find the game window ID
            result = await conn.run(
                f"DISPLAY={DISPLAY} xdotool search --name 'Oxygen Not Included'",
                check=False
            )
            if result.exit_status != 0 or not result.stdout.strip():
                return False  # game not running or window not found

            window_id = result.stdout.strip().splitlines()[-1]
            await conn.run(
                f"DISPLAY={DISPLAY} xdotool key --window {window_id} alt+F4",
                check=False
            )
            await asyncio.sleep(5)  # give it time to close
            return True
        except Exception:
            return False

    async def _kill_game(self, conn) -> None:
        """Force-kill the game process as fallback."""
        await conn.run(f"pkill -SIGTERM -f {GAME_PROCESS}", check=False)
        await asyncio.sleep(3)
        # SIGKILL if still alive
        await conn.run(f"pkill -SIGKILL -f {GAME_PROCESS}", check=False)

    async def _wait_for_process_death(self, conn, timeout_s: int = 15) -> None:
        """Poll until game process is confirmed dead."""
        for _ in range(timeout_s):
            result = await conn.run(
                f"pgrep -f {GAME_PROCESS}", check=False
            )
            if result.exit_status != 0:
                return  # process gone
            await asyncio.sleep(1)
        # Timeout — force kill
        await self._kill_game(conn)

    async def _launch_game(self, conn) -> None:
        """Launch ONI via Steam URL. Fire-and-forget — game takes time to load."""
        await conn.run(
            f"DISPLAY={DISPLAY} steam steam://rungameid/457140 &",
            check=False
        )
        # Steam needs time to start the game. Don't poll here —
        # _wait_for_bridge_ready() handles readiness detection.
        await asyncio.sleep(5)  # brief pause before polling begins

    async def _wait_for_bridge_ready(
        self, timeout_s: int = 120, poll_interval_s: float = 2.0
    ) -> None:
        """
        Poll TCP port 9999 on game host until the ONIBridge mod is accepting
        connections and sending valid state messages. This is the definitive
        'game is ready' signal — no fragile timing assumptions needed.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(GAME_HOST, BRIDGE_PORT),
                    timeout=3.0
                )
                # Connection established — wait for first state message
                line = await asyncio.wait_for(reader.readline(), timeout=10.0)
                writer.close()
                if b'"type": "state"' in line or b'"cycle"' in line:
                    return  # game is live
            except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
                pass
            await asyncio.sleep(poll_interval_s)
        raise TimeoutError(
            f"Bridge did not come up within {timeout_s}s after game launch"
        )
```

### Integration with `runner.py`

```python
# In Runner.__init__:
self._reloader = EpisodeReloader(save_path=CANONICAL_SAVE)

# In Runner._episode_loop, after episode end detected:
async def _on_episode_end(self, reason: str):
    self._log_episode(reason)
    result = await self._reloader.reset_episode()
    if result.success:
        logger.info(f"Episode reset in {result.elapsed_s:.1f}s — reconnecting")
        await self._reconnect()
    else:
        logger.error(f"Episode reset failed: {result.error}")
        raise RuntimeError("Reload failed — manual intervention required")
```

---

## Part 3: Canonical Save Setup

**Confirmed save file format (Spaced Out DLC):**
Each colony is stored as a directory containing `{colony-name}.sav` and
`{colony-name}.png`. Example from the live game host:
```
save_files/
  Bob-CV1/
    Bob-CV1.sav     ← 1.3MB actual save
    Bob-CV1.png     ← 1.7MB preview screenshot
    auto_save/      ← rolling auto-saves
  Bob-Colony/
    ...
```

**Creating the canonical training save:**

1. Start a fresh game on the canonical seed: `v-sndst-c-1427943156-0-1a-j3et5`
2. When naming the colony, use exactly: **`training-start`**
   (this makes the path predictable: `save_files/training-start/training-start.sav`)
3. Let it reach cycle 1 with dupes spawned — do not issue any commands
4. Save immediately (`Ctrl+S`)
5. Verify on the Linux desktop:
   ```bash
   ls -lh ~/.config/unity3d/Klei/Oxygen\ Not\ Included/save_files/training-start/
   # Expected:
   # training-start.sav
   # training-start.png
   ```
6. `CANONICAL_SAVE` in `reload.py` is pre-set to this path — no update needed if
   the colony is named exactly `training-start`

**Why a fixed colony name matters:** ONI auto-saves inside the colony directory using
the colony name as the filename. Using a predictable colony name means `reload.py`
never needs updating — the path is deterministic from day one.

---

## Reset Timing Estimates

| Phase | Duration |
|-------|----------|
| Graceful quit (Alt+F4) | 3–8s |
| Process death confirmation | 1–3s |
| Steam game launch | 20–40s |
| Mod load + autoload save | 5–10s |
| Bridge ready detection | 2–5s |
| **Total per reset** | **~35–65s** |

At 60s average reset: **~1,440 episode slots per day** (24h ÷ 60s). Even at 50%
utilization (training time, breaks): **~720 episodes per day** vs. 2–4 manually.

---

## Error Handling

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Game process not found at quit | `xdotool search` returns empty | Skip to `_kill_game` |
| Process refuses to die | Timeout in `_wait_for_process_death` | SIGKILL fallback |
| Steam doesn't launch game | Bridge timeout (120s) | Log error, raise — requires manual check |
| Save file missing | `AutoloadConfig` logs warning, game loads to main menu | Bridge connects but no state — runner detects and raises |
| autoload.txt write fails | SSH exception | Episode reset fails, logged |

---

## Dependencies

- `asyncssh` — SSH from Python (`pip install asyncssh`)
- `xdotool` — already installed on game host (confirmed working for settings button)
- Steam running on game host — assumed always running; add a Steam start step if needed

---

## Files Changed

| File | Change |
|------|--------|
| `mod/ONIBridge/src/AutoloadConfig.cs` | New file |
| `mod/ONIBridge/src/GameTickPatch.cs` | Add `TryAutoload()` call, one-shot guard |
| `src/agent/reload.py` | New file — `EpisodeReloader` class |
| `src/agent/runner.py` | Instantiate reloader, call on episode end |
| `requirements.txt` | Add `asyncssh` |

---

## Out of Scope

- Xvfb / virtual display (game requires physical GPU rendering)
- Headless mode (not supported by ONI)
- Multi-instance parallel training (single display constraint; future work with
  multiple physical machines)
- Save file versioning / rotation (always reload canonical start save)
