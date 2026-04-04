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
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.agent.client import BridgeClient
from src.agent.llm import LLMAgent, DEFAULT_ENDPOINT, DEFAULT_MODEL
from src.agent.perimeter import SpatialLedger
from src.agent.protocol import build_abandon_perimeter, build_no_op
from src.agent.reload import CANONICAL_SAVE, EpisodeReloader
from src.agent.reward import RewardCalculator, format_colony_health

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

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

# Colony type policy: enforced as a hard constraint on accept_print actions.
# "organic_only" | "bionic_only" | "mixed"
COLONY_TYPE_POLICY = "organic_only"

# Episode configuration
EPISODE_MAX_CYCLE = 100          # end episode if cycle reaches this
EPISODE_WIN_PHASE = 0            # current win condition phase
EMPTY_COLONY_TICKS_THRESHOLD = 3 # consecutive empty-duplicants ticks = loss

# Win condition thresholds by phase
WIN_CONDITIONS: dict[int, dict] = {
    0: {"survive_cycles": 3,  "require_no_deaths": True},   # Phase 0: smoke test
    1: {"survive_cycles": 10, "require_no_deaths": False},   # Phase 1: cycle 10
    2: {"survive_cycles": 25, "require_spom": True},         # Phase 2: SPOM active
    3: {"survive_cycles": 50, "require_dupes": 5},           # Phase 3: 5+ dupes
    4: {"survive_cycles": 100},                              # Phase 4: full run
}

# Canonical training seed
CANONICAL_SEED = "v-sndst-c-1427943156-0-1a-j3et5"


def _should_accept_offer(offer: dict) -> bool:
    """Enforce COLONY_TYPE_POLICY before sending accept_print to game."""
    if offer.get("type") == "care_package":
        return True  # always accept resource packages
    if offer.get("type") == "duplicant":
        if COLONY_TYPE_POLICY == "organic_only":
            return offer.get("subtype") == "organic"
        if COLONY_TYPE_POLICY == "bionic_only":
            return offer.get("subtype") == "bionic"
    return True  # mixed: accept anything


def _check_win_condition(state: dict, episode_record: "Any") -> "str | None":
    """
    Returns end condition string if episode should end, None to continue.
    End conditions: "win", "loss", "cycle_limit"
    """
    phase = EPISODE_WIN_PHASE
    win_cond = WIN_CONDITIONS.get(phase, {})
    cycle = state.get("cycle", 0)
    dupes = state.get("duplicants", [])
    buildings = {b["type"] for b in state.get("buildings", [])}

    # Loss: check is done in run() by tracking consecutive empty ticks

    # Win: check phase conditions
    survive_cycles = win_cond.get("survive_cycles", 10)
    if cycle >= survive_cycles:
        if win_cond.get("require_no_deaths") and episode_record.total_deaths > 0:
            return None  # need no deaths — keep going until end or loss
        if win_cond.get("require_spom"):
            if "Electrolyzer" not in buildings or "HydrogenGenerator" not in buildings:
                return None  # SPOM not active yet
        if win_cond.get("require_dupes", 0) > 0:
            if len(dupes) < win_cond["require_dupes"]:
                return None  # not enough dupes yet
        return "win"

    # Neutral: hit max cycle
    if cycle >= EPISODE_MAX_CYCLE:
        return "cycle_limit"

    return None


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
        self.pending_action: dict | None = None  # last action sent to game, cleared each cycle

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


async def run(
    host: str,
    port: int,
    endpoint_url: str,
    model: str,
    api_key: str,
    episode_log: Path | None,
    reloader: EpisodeReloader | None = None,
) -> None:
    agent = LLMAgent(endpoint_url=endpoint_url, model=model, api_key=api_key)

    while True:
        client = BridgeClient(host=host, port=port)
        ledger = SpatialLedger()
        reward_calc = RewardCalculator(ledger=ledger)

        log_entries: list[dict] = []
        last_ai_action: dict | None = None
        last_ai_action_tick: int = 0
        last_cycle: int = 0
        empty_colony_ticks: int = 0  # consecutive ticks with no dupes
        episode_end_condition: str | None = None
        episode_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        logger.info("Connecting to ONI bridge at %s:%d ...", host, port)
        await client.connect()
        logger.info("Connected. Starting agent loop. Relay on ws://0.0.0.0:%d/ws", RELAY_PORT)

        tick = 0
        async for state in client.state_stream():
            tick += 1
            cycle = state.cycle
            resources = state.data.get("resources", {})
            alerts = state.data.get("alerts", [])

            # Update spatial ledger each tick
            ledger.on_state(state.data)

            # Auto-complete: perimeter hit 100% — send abandon to clean up mod side
            if ledger.autocomplete_pending:
                ledger.clear_autocomplete()
                logger.info("Perimeter auto-complete — sending abandon_perimeter")
                await client.send_action(build_abandon_perimeter())
                relay.pending_action = build_abandon_perimeter()

            # Reward calculation
            tick_reward = reward_calc.tick(state.data)
            episode_reward = reward_calc.episode_total
            obligations = reward_calc.open_obligations()

            # Episode lifecycle: empty colony check
            if not state.data.get("duplicants"):
                empty_colony_ticks += 1
            else:
                empty_colony_ticks = 0

            if empty_colony_ticks >= EMPTY_COLONY_TICKS_THRESHOLD:
                episode_end_condition = "loss"
                logger.warning("All dupes dead for %d ticks — episode LOSS", empty_colony_ticks)

            # Win condition check
            if episode_end_condition is None:
                episode_end_condition = _check_win_condition(
                    state.data, reward_calc.episode_record
                )

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

            # Reset dedup and pending action on new cycle
            if cycle != last_cycle:
                last_ai_action = None
                relay.pending_action = None
                last_cycle = cycle

            # Check for a manual action from the dashboard first; fall back to AI
            manual = relay.get_manual_action_nowait()
            if manual:
                action = manual
                last_ai_action = None  # reset dedup on manual override
                logger.info("  -> manual action from dashboard: %s", action)
            else:
                colony_health = format_colony_health(
                    state.data, tick_reward, episode_reward, obligations
                )
                candidate = agent.decide(
                    state.data,
                    pending_action=relay.pending_action,
                    ledger_context=ledger.format_context(),
                    colony_health=colony_health,
                )
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
            relay.pending_action = action

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
                    "episode_id": episode_id,
                    "tick": tick,
                    "cycle": cycle,
                    "state": state.data,
                    "action": action,
                    "reward": tick_reward,
                    "ledger_snapshot": ledger.to_dict(),
                })

        logger.info("Bridge closed after %d ticks.", tick)

        # Compute final outcome reward
        outcome_reward = reward_calc.episode_end()
        ep = reward_calc.episode_record

        logger.info(
            "Episode %s done: end=%s ticks=%d cycles=%d deaths=%d reward=%.1f outcome=%.1f",
            episode_id,
            episode_end_condition or "disconnect",
            tick,
            ep.final_cycle,
            ep.total_deaths,
            ep.total_reward,
            outcome_reward,
        )

        if episode_log is not None and log_entries:
            # Default path: data/episodes/YYYYMMDD-HHMMSS-<seed>.jsonl
            if episode_log == Path("auto"):
                episode_log = Path("data/episodes") / f"{episode_id}-{CANONICAL_SEED[:12]}.jsonl"

            episode_log.parent.mkdir(parents=True, exist_ok=True)

            # Write JSONL: one JSON object per line
            with open(episode_log, "w") as f:
                # First line: episode summary
                summary = {
                    "episode_id": episode_id,
                    "seed": CANONICAL_SEED,
                    "end_condition": episode_end_condition or "disconnect",
                    "start_cycle": ep.start_cycle,
                    "final_cycle": ep.final_cycle,
                    "total_ticks": tick,
                    "total_deaths": ep.total_deaths,
                    "total_reward": ep.total_reward,
                    "outcome_reward": outcome_reward,
                    "milestones": ep.milestones,
                    "agent_stats": agent.stats,
                }
                f.write(json.dumps(summary) + "\n")
                for entry in log_entries:
                    f.write(json.dumps(entry) + "\n")
            logger.info("Episode log saved to %s (%d ticks)", episode_log, len(log_entries))

        if reloader is None:
            # No auto-reload: exit after the first episode
            break

        logger.info("Auto-reload enabled — resetting episode...")
        result = await reloader.reset_episode()
        if result.success:
            logger.info("Episode reset in %.1fs — reconnecting", result.elapsed_s)
            # Loop back to reconnect and start the next episode
        else:
            logger.error("Episode reset failed: %s — stopping", result.error)
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="ONI AI agent runner + WebSocket relay")
    parser.add_argument("--host",    default="10.0.0.10",  help="Game bridge host")
    parser.add_argument("--port",    type=int, default=9999, help="Game bridge port")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT,
                        help="WebSocket relay port for dashboard (default 8182)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help="OpenAI-compatible LLM endpoint URL")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name to use")
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY", ""),
                        help="API key (optional for local endpoints)")
    parser.add_argument("--log-episode", default=None,
                        help="Path to write episode JSONL log, or 'auto' for data/episodes/ auto-naming")
    parser.add_argument("--save", default=CANONICAL_SAVE, help="Save path for episode resets")
    parser.add_argument("--auto-reload", action="store_true", help="Enable automatic episode reloading")
    args = parser.parse_args()

    episode_log = Path(args.log_episode) if args.log_episode else None
    if args.log_episode == "auto":
        episode_log = Path("auto")  # sentinel value; actual path set inside run()

    global _game_host
    _game_host = args.host

    reloader: EpisodeReloader | None = (
        EpisodeReloader(save_path=args.save) if args.auto_reload else None
    )

    async def _run_all() -> None:
        # Start the uvicorn relay server as a background task
        config = uvicorn.Config(app, host="0.0.0.0", port=args.relay_port,
                                log_level="warning")
        server = uvicorn.Server(config)
        relay_task = asyncio.create_task(server.serve())
        agent_task = asyncio.create_task(
            run(args.host, args.port, args.endpoint, args.model, args.api_key, episode_log, reloader)
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
