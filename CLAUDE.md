# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ONI-AI is an AI agent for playing Oxygen Not Included (ONI). The current approach uses a live
**Gemini LLM agent** connected to the running game via a C# Harmony mod (ONIBridge). A longer-term
RL pipeline (Phases 1–5) is scaffolded but not the active focus.

### Current Status (2026-04-04)

| Component | Status |
|-----------|--------|
| ONIBridge C# mod (TCP bridge) | Live — deployed on Linux desktop |
| Gemini agent (Python) | Live — first session ran 2026-04-04 |
| Dashboard (FastAPI + WebSocket) | Live — speed controls, settings button, live state |
| Wiki tool calling | Complete — SQLite FTS5, Gemini function calling |
| Grid vision (tile window) | Complete — 64×64 tile window in state payload |
| RL pipeline (Phases 1–5) | Scaffolded, not active focus |

### Known Issues (from first live session)
- Agent loses spatial reasoning after a few actions — tile window needs better prompt framing
- Stress values read >1.0 (StressMonitor.stress.value not clamped)
- Food kcal reads ~16M (Edible.Calories unit mismatch — likely grams not kcal)

See `docs/session-logs/` for session notes.

## Infrastructure

| Machine | IP | Role | SSH |
|---------|----|------|-----|
| Mac Mini M4 Pro | 10.0.0.210 | Dev machine (YOU ARE HERE) | localhost |
| Linux Desktop | 10.0.0.10 | Game host (ONI + ONIBridge) | `ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 myroproductions@10.0.0.10` |
| DGX Spark | 10.0.0.69 | AI compute (future training) | `ssh dgx1-ssh` |

**ONI mod path on Linux desktop:**
`~/.config/unity3d/Klei/Oxygen Not Included/mods/Dev/ONIBridge/ONIBridge.dll`

**Game log:**
`~/.config/unity3d/Klei/Oxygen Not Included/Player.log`

## Running the Agent

```bash
# 1. Build and deploy the mod (from Mac)
cd mod/ONIBridge && dotnet build
scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 \
  bin/Debug/net471/ONIBridge.dll \
  "myroproductions@10.0.0.10:/home/myroproductions/.config/unity3d/Klei/Oxygen Not Included/mods/Dev/ONIBridge/ONIBridge.dll"

# 2. Start the runner (from Mac, game must be running on Linux desktop)
GOOGLE_API_KEY=<key> python3 -m src.agent.runner --host 10.0.0.10

# 3. Start the dashboard (separate terminal)
python3 examples/dashboard/server.py
# Open http://localhost:8181

# 4. Build wiki DB (one-time)
python3 scripts/build_wiki_db.py
```

## Architecture

### ONIBridge (C# Harmony Mod)

`mod/ONIBridge/src/`
- `ONIBridgeMod.cs` — Harmony entry point, registers patches
- `BridgeServer.cs` — TCP server on port 9999, newline-delimited JSON
- `BridgeTicker.cs` — MonoBehaviour coroutine, fires every 1s
- `GameTickPatch.cs` — Harmony patch on SimEveryTick, drains action queue
- `StateSerializer.cs` — Serializes game state → JSON (cycle, resources, duplicants, buildings, alerts, tiles)
- `ActionExecutor.cs` — Executes AI actions (place_building, dig, cancel_dig, set_priority, set_speed, no_op)
- `ActionCommand.cs` — Deserialized action payload

### Python Agent

`src/agent/`
- `runner.py` — Main loop: owns TCP connection, runs Gemini, serves WebSocket relay on :8182
- `llm.py` — `GeminiAgent`: formats state prompt, multi-turn wiki tool calling, parses action JSON
- `protocol.py` — Message types, `VALID_ACTIONS`, `build_action()`, `parse_state_message()`
- `client.py` — Async TCP client, `state_stream()` generator

### Dashboard

`examples/dashboard/`
- `index.html` — Single-page dashboard: live state, building placement drawer, speed controls, settings button
- `server.py` — FastAPI server on :8181, subscribes to runner relay, proxies actions back

### Wiki

`scripts/build_wiki_db.py` — One-time scraper → `data/wiki.db` (SQLite FTS5, gitignored)

### Protocol (TCP, newline-delimited JSON)

**Game → Agent (state):**
```json
{
  "type": "state",
  "data": {
    "cycle": 1, "time": 100.3,
    "resources": {"oxygen_kg": 0.0, "water_kg": 0.0, "food_kcal": 0.0, "power_kw": 0.0, "co2_kg": 0.0},
    "duplicants": [{"id": -126192, "name": "Lindsay", "x": 132, "y": 203, "stress": 0.5, "health": 100.0, "current_task": "Dig"}],
    "buildings": [{"type": "Tile", "x": 127, "y": 202, "operational": true}],
    "alerts": [],
    "tiles": {"x": 109, "y": 187, "w": 64, "h": 64, "data": [["Sandstone", 1800.0], ...]}
  }
}
```

**Agent → Game (action):**
```json
{"type": "action", "action": "dig", "cell_x": 115, "cell_y": 202}
{"type": "action", "action": "place_building", "building_id": "Bed", "cell_x": 116, "cell_y": 201}
{"type": "action", "action": "set_speed", "speed": 1}
{"type": "action", "action": "no_op"}
```

**Dashboard UI action (WebSocket, not forwarded to game TCP):**
```json
{"type": "ui_action", "action": "open_settings"}
```

## Commands

### Testing

```bash
# Agent tests (fast, no game required)
pytest tests/agent/ -v

# All tests (requires pandas for integration tests)
pytest tests/ -v
```

### Build Mod

```bash
cd mod/ONIBridge
dotnet build
# Output: bin/Debug/net471/ONIBridge.dll
```

### Code Quality

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Code Conventions

- **Type hints** required on all function signatures
- **Dataclasses** for structured data
- **Google-style docstrings**
- **Import order**: stdlib → third-party → project → relative
- **Naming**: `snake_case` for Python, `PascalCase` for C#

## Specifications & Plans

- `docs/superpowers/specs/` — design specs for each feature
- `docs/superpowers/plans/` — implementation plans
- `docs/session-logs/` — notes from live agent sessions
- `.kiro/specs/oni-ai/` — original RL pipeline specs (Phases 1–5, lower priority)
