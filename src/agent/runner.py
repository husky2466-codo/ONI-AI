# src/agent/runner.py
"""
ONI AI Agent runner with WebSocket relay.

Owns the single TCP connection to the game bridge, runs the Gemini AI loop,
and serves a WebSocket relay on port 8182 so the dashboard can subscribe
without competing for the game socket.

WebSocket relay protocol (JSON messages):
  Server → clients:  {"type": "state", "data": {...}}
                     {"type": "ack", "log": [...]}
  Client → server:   {"type": "action", "action": "...", ...params}

Usage:
    python3 -m src.agent.runner --host 10.0.0.10 --api-key <key>

    # Or with GOOGLE_API_KEY env var set:
    python3 -m src.agent.runner --host 10.0.0.10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.agent.client import BridgeClient
from src.agent.llm import GeminiAgent
from src.agent.protocol import build_no_op

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("oni.runner")

# xdotool coordinates for the ONI settings cog (1920x988 windowed, window at 0,32)
_COG_X = 1893
_COG_Y = 88
_SSH_USER = "myroproductions"
_SSH_KEY = str(Path.home() / ".ssh" / "id_ed25519")


async def open_settings_via_xdotool(host: str) -> bool:
    """SSH to the game host and click the in-game settings cog button."""
    # Single shell string passed as one argument — no user input interpolated into it,
    # all values are module-level constants. The host is passed as a separate SSH arg.
    xdotool_cmd = (
        f"export DISPLAY=:0 && "
        f"xdotool mousemove 960 500 click 1 && "
        f"sleep 0.3 && "
        f"xdotool mousemove {_COG_X} {_COG_Y} click 1"
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=no",
            "-i", _SSH_KEY,
            f"{_SSH_USER}@{host}",
            xdotool_cmd,
        )
        rc = await proc.wait()
        if rc == 0:
            logger.info("Settings cog clicked via xdotool on %s", host)
            return True
        logger.warning("xdotool SSH exited with code %d", rc)
        return False
    except Exception as e:
        logger.warning("open_settings_via_xdotool failed: %s", e)
        return False

RELAY_PORT = 8182


# ---------------------------------------------------------------------------
# WebSocket relay — fan-out to all connected dashboard clients
# ---------------------------------------------------------------------------

class RelayManager:
    def __init__(self) -> None:
        self._clients: list[WebSocket] = []
        self._manual_actions: asyncio.Queue[dict] = asyncio.Queue()
        self.log: list[str] = []
        self.last_state: dict = {}

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)
        # Send current state immediately on connect
        if self.last_state:
            await ws.send_json({"type": "state", "data": self.last_state})
        if self.log:
            await ws.send_json({"type": "ack", "log": self.log[-20:]})

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, data: dict) -> None:
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.remove(ws)

    def append_log(self, msg: str) -> None:
        self.log.append(msg)
        if len(self.log) > 50:
            self.log.pop(0)

    async def enqueue_manual_action(self, action: dict) -> None:
        await self._manual_actions.put(action)

    def get_manual_action_nowait(self) -> dict | None:
        try:
            return self._manual_actions.get_nowait()
        except asyncio.QueueEmpty:
            return None


relay = RelayManager()
app = FastAPI()
_game_host = "10.0.0.10"  # set at startup from --host arg


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await relay.connect(ws)
    logger.info("Dashboard client connected")
    try:
        while True:
            msg = await ws.receive_json()
            logger.info("Dashboard action: %s", msg)
            if msg.get("type") == "ui_action" and msg.get("action") == "open_settings":
                # Fire xdotool click on the game host — does not go through TCP bridge
                asyncio.create_task(open_settings_via_xdotool(_game_host))
            elif msg.get("type") == "action" and msg.get("action"):
                await relay.enqueue_manual_action(msg)
    except WebSocketDisconnect:
        relay.disconnect(ws)
        logger.info("Dashboard client disconnected")
    except Exception as e:
        relay.disconnect(ws)
        logger.warning("Dashboard WS error: %s", e)


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

_DEDUP_TTL = 5  # suppress identical AI actions for this many ticks


async def run(host: str, port: int, api_key: str, episode_log: Path | None) -> None:
    client = BridgeClient(host=host, port=port)
    agent = GeminiAgent(api_key=api_key)

    log_entries: list[dict] = []
    last_ai_action: dict | None = None
    last_ai_action_tick: int = 0
    last_cycle: int = 0

    logger.info("Connecting to ONI bridge at %s:%d ...", host, port)
    await client.connect()
    logger.info("Connected. Starting agent loop. Relay on ws://0.0.0.0:%d/ws", RELAY_PORT)

    tick = 0
    async for state in client.state_stream():
        tick += 1
        cycle = state.cycle
        resources = state.data.get("resources", {})
        alerts = state.data.get("alerts", [])

        # Update relay state and broadcast to dashboard subscribers
        relay.last_state = state.data
        await relay.broadcast({
            "type": "state",
            "data": state.data,
            "agent": agent.stats,
            "tick": tick,
        })

        logger.info(
            "[tick %d | cycle %d] O2=%.2fkg food=%.0fkcal power=%.2fkW alerts=%d",
            tick, cycle,
            resources.get("oxygen_kg", 0),
            resources.get("food_kcal", resources.get("food_kcal_today", 0)),
            resources.get("power_kw", 0),
            len(alerts),
        )
        if alerts:
            for a in alerts:
                logger.warning("  ALERT: %s", a)

        # Reset dedup on new cycle so AI can issue fresh commands each cycle
        if cycle != last_cycle:
            last_ai_action = None
            last_cycle = cycle

        # Check for a manual action from the dashboard first; fall back to AI
        manual = relay.get_manual_action_nowait()
        if manual:
            action = manual
            last_ai_action = None  # reset dedup on manual override
            logger.info("  -> manual action from dashboard: %s", action)
        else:
            candidate = agent.decide(state.data)
            # Suppress repeated identical non-no_op AI actions for _DEDUP_TTL ticks
            if (candidate == last_ai_action
                    and candidate.get("action") != "no_op"
                    and (tick - last_ai_action_tick) < _DEDUP_TTL):
                logger.info("  -> dedup: suppressing repeat %s (sent %d ticks ago)",
                            candidate.get("action"), tick - last_ai_action_tick)
                action = build_no_op()
            else:
                action = candidate
                last_ai_action = candidate
                last_ai_action_tick = tick
            logger.info("  -> AI action: %s", action)

        await client.send_action(action)

        # Read ACK (non-blocking peek — ACK arrives on the same TCP stream,
        # but state_stream() only yields state messages.  We handle the raw
        # ACK inside BridgeClient by extending state_stream to pass acks through.)
        # For now, log optimistically and let the next state confirm success.
        ack_log = f"[{tick}] {action.get('action')}" + (
            f" {action.get('building_id','')} @({action.get('cell_x','')},{action.get('cell_y','')})"
            if action.get('action') == 'place_building' else
            f" @({action.get('cell_x','')},{action.get('cell_y','')})"
            if action.get('cell_x') is not None else ""
        )
        relay.append_log(ack_log)
        await relay.broadcast({"type": "ack", "log": relay.log[-20:], "last_action": action})

        if episode_log is not None:
            log_entries.append({
                "tick": tick,
                "cycle": cycle,
                "state": state.data,
                "action": action,
            })

    logger.info("Bridge closed after %d ticks.", tick)

    if episode_log is not None and log_entries:
        episode_log.parent.mkdir(parents=True, exist_ok=True)
        with open(episode_log, "w") as f:
            json.dump(log_entries, f, indent=2)
        logger.info("Episode log saved to %s", episode_log)


def main() -> None:
    parser = argparse.ArgumentParser(description="ONI Gemini agent runner + WebSocket relay")
    parser.add_argument("--host",    default="10.0.0.10",  help="Game bridge host")
    parser.add_argument("--port",    type=int, default=9999, help="Game bridge port")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT,
                        help="WebSocket relay port for dashboard (default 8182)")
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY", ""),
                        help="Google API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--log-episode", default=None,
                        help="Path to write episode JSON log (optional)")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key is required (or set GOOGLE_API_KEY env var)")

    episode_log = Path(args.log_episode) if args.log_episode else None

    global _game_host
    _game_host = args.host

    async def _run_all() -> None:
        # Start the uvicorn relay server as a background task
        config = uvicorn.Config(app, host="0.0.0.0", port=args.relay_port,
                                log_level="warning")
        server = uvicorn.Server(config)
        relay_task = asyncio.create_task(server.serve())
        agent_task = asyncio.create_task(
            run(args.host, args.port, args.api_key, episode_log)
        )
        # Run both; if agent loop exits, cancel the relay
        done, pending = await asyncio.wait(
            [relay_task, agent_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
