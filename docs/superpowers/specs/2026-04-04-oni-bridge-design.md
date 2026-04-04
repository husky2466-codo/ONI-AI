# ONI Bridge — Design Spec
**Date:** 2026-04-04
**Status:** Approved for implementation
**Project:** ONI-AI (`/Volumes/DevDrive-M4Pro/Projects/ONI-AI`)

---

## Overview

Build a system where an AI agent autonomously plays Oxygen Not Included (ONI) in real time. The AI reads live game state, makes decisions via LLM inference, and issues action commands that the game executes — no human in the loop.

This goes one step beyond the reference mod (video by creator with Gemini API) which only provided advisory output to a human operator. Our system closes the loop: the AI is the player.

The long-term goal is to feed game episodes back into a training pipeline on the DGX Sparks, progressively improving the agent's colony management capability.

---

## Infrastructure

| Role | Machine | Purpose |
|---|---|---|
| Game host | Ubuntu Desktop (RTX 4070, 30GB RAM) | Runs ONI + C# Harmony mod |
| AI inference | DGX Spark 1 + DGX Spark 2 | vLLM serving + BobSpark-APIs |
| Development | Mac Mini M4 Pro | C# mod dev (Rider), Python agent dev, dashboard dev |
| Dashboard | Mac Mini M4 Pro | React dashboard served locally, observes live sessions |

---

## Architecture

```
┌─────────────────────────────────────────┐
│           Ubuntu Desktop                │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │     Oxygen Not Included          │   │
│  │                                  │   │
│  │  ┌─────────────────────────┐     │   │
│  │  │  ONIBridge C# Mod       │     │   │
│  │  │  - Harmony hooks        │     │   │
│  │  │  - TCP server :9999     │     │   │
│  │  │  - State serializer     │     │   │
│  │  │  - Action executor      │     │   │
│  │  └────────────┬────────────┘     │   │
│  └───────────────┼──────────────────┘   │
└──────────────────┼──────────────────────┘
                   │ TCP (newline-delimited JSON)
                   │
┌──────────────────┼──────────────────────┐
│              DGX Spark 1                │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │     ONI Agent (Python)           │   │
│  │  - TCP client → game socket      │   │
│  │  - State parser                  │   │
│  │  - LLM prompt builder            │   │
│  │  - Action decoder                │   │
│  │  - Episode recorder              │   │
│  └──────┬───────────────────────────┘   │
│         │ HTTP (OpenAI-compat API)       │
│  ┌──────▼───────────────────────────┐   │
│  │   vLLM (BobSpark-APIs)           │   │
│  │   - Qwen2.5-Coder-32B or         │   │
│  │     Llama 3.3-70B                │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
                   │ WebSocket (metrics relay)
                   │
┌──────────────────┼──────────────────────┐
│           Mac Mini M4 Pro               │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │   ONI Dashboard (React/Vite)     │   │
│  │   - Live colony metrics          │   │
│  │   - AI terminal (reasoning log)  │   │
│  │   - Action feed                  │   │
│  │   - Episode history              │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## Components

### 1. ONIBridge — C# Harmony Mod

**Location:** `mod/ONIBridge/`
**Runs on:** Ubuntu Desktop, inside ONI game process
**Language:** C# targeting net471 (Mono, same as ONI)

Responsibilities:
- Open a TCP server on port 9999 on a background thread at mod load
- Hook `Game.Update` via Harmony postfix to drain action queue and send state snapshots
- Serialize game state every N ticks: cycle, time, resources, duplicants, buildings, alerts, grid summary
- Receive newline-delimited JSON action commands from the AI agent
- Execute actions on the main thread by calling ONI's internal C# methods directly

Key files:
- `ONIBridgeMod.cs` — `UserMod2` entry point
- `BridgeServer.cs` — TCP server, background thread, concurrent action queue
- `GameTickPatch.cs` — Harmony `Game.Update` postfix hook
- `StateSerializer.cs` — game state → JSON-safe object
- `ActionExecutor.cs` — JSON command → ONI internal API calls
- `ActionCommand.cs` — command schema

Protocol (newline-delimited JSON over TCP):
```
Game → Agent:  { "type": "state", "cycle": 5, "data": { ... } }
Agent → Game:  { "type": "action", "action": "place_building", "building_id": "OxygenDiffuser", "cell_x": 12, "cell_y": 8 }
Game → Agent:  { "type": "ack", "success": true, "action": "place_building" }
```

Action types (Phase 1):
- `place_building` — `building_id`, `cell_x`, `cell_y`
- `dig` — `cell_x`, `cell_y`
- `cancel` — `cell_x`, `cell_y`
- `set_priority` — `cell_x`, `cell_y`, `priority` (1-9)
- `no_op` — do nothing this tick

State snapshot fields (Phase 1):
- `cycle`, `time`
- `resources` — oxygen_kg, water_kg, food_kcal, power_kw, co2_kg
- `duplicants[]` — id, name, x, y, stress, health, current_task
- `alerts[]` — active warning strings from colony diagnostics
- `buildings[]` — type, x, y, operational status
- `grid_summary` — min/max oxygen, avg temperature by zone

### 2. ONI Agent — Python Client

**Location:** `src/agent/`
**Runs on:** DGX Spark 1
**Language:** Python 3.11+

Responsibilities:
- Maintain TCP connection to the mod bridge, reconnect on drop
- Parse incoming state snapshots
- Build structured prompts from game state for LLM inference
- Decode LLM output into `ActionCommand` JSON
- Send actions back over the socket
- Record `(state, prompt, response, action, ack)` tuples to episode log for training pipeline

LLM integration:
- Calls vLLM via OpenAI-compatible API (`http://10.0.0.69:<port>/v1/chat/completions`)
- Two-model pattern (same as reference mod, but both closed-loop):
  - **Tactical model** — called every N ticks, handles immediate survival needs
  - **Strategic model** — called every M cycles, sets longer-horizon goals
- Tactical model receives last 3 state snapshots + current strategic goal
- Strategic model receives cycle summary + resource trends

Prompt structure:
```
System: You are an autonomous colony manager for Oxygen Not Included.
        Output ONLY valid JSON matching the ActionCommand schema.
        Current strategic goal: {goal}

User:   Colony state at cycle {cycle}:
        {state_summary}
        Recent alerts: {alerts}
        Available actions: {action_list}
        What is your next action?
```

### 3. Metrics Relay — Python Side-Car

**Location:** `src/relay/`
**Runs on:** DGX Spark 1, alongside the agent

Responsibilities:
- Subscribes to state snapshots from the bridge
- Forwards metrics over WebSocket to the dashboard on Mac Mini
- Buffers episode data to disk for training pipeline ingestion
- Exposes `/api/episode` REST endpoint for dashboard to query history

### 4. ONI Dashboard — React/Vite

**Location:** `dashboard/`
**Runs on:** Mac Mini M4 Pro
**Stack:** React, Vite, Tailwind, WebSocket — same stack as Molt-Government

Pages:
- **Live Colony** — real-time metrics: oxygen, food, power, stress gauges; duplicant roster; active alerts
- **AI Terminal** — scrolling log of AI reasoning, current strategic goal, last action issued; color-coded status (green/amber/red) matching colony health
- **Action Feed** — timestamped stream of every action the AI issued and whether it succeeded
- **Episode History** — past runs: survival cycles, cause of death/win, reward curve

Visual style: dark theme, monospace stats, color-coded status indicators — consistent with Molt-Government's capitol theme adapted to a sci-fi/industrial aesthetic matching ONI's visual language.

---

## Data Flow — Single Game Tick

```
1. Game.Update() fires (main thread)
2. GameTickPatch.Postfix() runs:
   a. BridgeServer.DrainActions() — executes any queued AI commands
   b. Every 10 ticks: StateSerializer.Serialize() → BridgeServer.SendState()
3. Agent receives state snapshot over TCP
4. Agent builds prompt, calls vLLM
5. Agent decodes response → ActionCommand JSON
6. Agent sends command over TCP
7. BridgeServer enqueues command
8. Next Game.Update() → DrainActions() executes it
9. Game sends ACK
10. Episode recorder logs (state, action, ack) tuple
```

---

## Training Pipeline (Phase 2)

Once the agent is playing correctly:

1. Episode recorder saves `(state, action, reward)` tuples per tick to JSONL
2. Reward function computed offline: survival cycles, oxygen coverage, duplicant stress, milestone completions
3. Dataset builder converts episodes to training format (SFT first, then RLHF/GRPO)
4. Fine-tuning job runs on DGX Spark 2 via existing BobSpark-APIs pipeline
5. Improved model checkpoint deployed back to vLLM on DGX Spark 1
6. Loop repeats

---

## Implementation Phases

### Phase 1 — Mod Bridge + Stub Agent (current)
- [ ] Copy game DLLs to `mod/ONIBridge/lib/`
- [ ] Build mod, verify it loads in ONI on Ubuntu
- [ ] Implement `StateSerializer` using real game APIs (decompile `Assembly-CSharp.dll` in Rider)
- [ ] Implement `ActionExecutor` for `place_building`, `dig`, `set_priority`
- [ ] Write Python TCP client that connects and prints state
- [ ] Verify round-trip: agent connects, receives state, sends `no_op`, receives ack

### Phase 2 — LLM Agent
- [ ] Build prompt templates for tactical and strategic models
- [ ] Connect to vLLM on DGX via BobSpark-APIs
- [ ] Implement action decoder with schema validation
- [ ] Run first autonomous session — colony survival as success metric
- [ ] Episode recorder writing JSONL

### Phase 3 — Dashboard
- [ ] Scaffold React/Vite project in `dashboard/`
- [ ] WebSocket relay from agent to dashboard
- [ ] Live Colony page
- [ ] AI Terminal page (scrolling reasoning log)
- [ ] Action Feed page

### Phase 4 — Training Loop
- [ ] Reward function definition
- [ ] Dataset builder for episode JSONL → training format
- [ ] Fine-tuning integration with DGX Spark 2
- [ ] Model versioning and eval benchmarks

---

## Key Decisions

**TCP over WebSocket for game bridge:** WebSocket requires an HTTP upgrade handshake that adds complexity inside the Mono runtime. Plain TCP with newline-delimited JSON is simpler, proven (ZTransport uses the same pattern), and sufficient for localhost communication.

**Newline-delimited JSON protocol:** Simple to implement in both C# and Python, debuggable with `nc` or `telnet`, no framing complexity.

**Two-model tactical/strategic split:** Matches the reference mod's proven architecture. Tactical model stays fast (small context, frequent calls). Strategic model gets richer context less often. Both output structured JSON, not free text.

**`Game.Update` hook over simulation tick hook:** `Game.Update` runs on the main Unity thread every frame, giving us a reliable place to drain the action queue safely. Simulation ticks run at a different rate and are not always on the main thread.

**Actions call internal C# methods directly:** Not screen capture, not simulated mouse clicks. `BuildingDef.TryPlace()`, dig chore creation, priority setting — same code the UI calls. Reliable, fast, survives UI changes.

---

## Open Questions

1. **Which vLLM model to start with?** Qwen2.5-Coder-32B (strong structured JSON output) vs Llama 3.3-70B (stronger reasoning). Recommend starting with Qwen2.5-Coder-32B for the action decoding reliability.
2. **State snapshot frequency:** 10 game updates = ~1 real second at normal speed. May need tuning based on how fast vLLM responds.
3. **Grid serialization:** Full 256x256 grid is too large to send every tick. Need a zone-summary or diff-based approach. TBD in Phase 1 implementation.
4. **Multiplayer mod compatibility:** The `onimp` multiplayer mod is not needed and should not be installed alongside ONIBridge — potential conflicts with game loop hooks.
