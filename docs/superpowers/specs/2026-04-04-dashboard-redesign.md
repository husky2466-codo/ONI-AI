# Dashboard Redesign — Design Spec

**Date:** 2026-04-04
**Status:** Approved — ready for implementation planning

---

## Overview

Replace the current single-page dashboard with a tabbed operator console. The dashboard is an
observability tool — the human operator watches what the AI is doing, not intervenes in it.
Config changes and connection management are the only operator-driven actions.

Base font: 13px (Courier New monospace). Dark theme matching current dashboard (#0d1117 background).

---

## Architecture

### Connection topology

```
Browser (dashboard :8181)
  └── FastAPI server (server.py)
        └── WebSocket relay (:8182) → runner.py
              └── TCP → ONIBridge (:9999) → ONI game
```

The dashboard subscribes to the runner relay for live state pushes. Actions flow back through
the same path. UI-only actions (open_settings) are type `ui_action` and not forwarded to the game.

### Layout structure

Every tab shares:
- **Status bar** (top) — connection chain status + runner state + last-updated timestamp
- **Tab bar** — 6 tabs
- **Tab body** — tab-specific content
- **Colony Health strip** (pinned above toolbar) — always-visible tick reward, episode total, survival count, open obligations
- **Toolbar** (bottom) — quick-action buttons + speed controls + Settings button

---

## Tabs

### Tab 1 — Colony

Two-column layout.

**Left column:**
- Game card: cycle, time, phase, speed indicator, game state (running/paused)
- Resources card: O2 kg, water kg, food kcal, power kW, CO2 kg — each with a thin progress bar scaled to soft thresholds
- Storage inventory: icon grid of stored resource types, click an icon to see kg amount

**Right column:**
- Alerts card: active alert list with severity badges
- Map/tile window card: 64×64 tile grid rendered as a compact colour-coded grid (tile type → colour), shows current window position (x,y)

---

### Tab 2 — Duplicants

Full-width scrollable list of dupe cards. One card per duplicant.

Each card shows:
- Name, type badge (organic / bionic), current task
- Stress bar, health bar
- Hunger bar (0=full, 1=starving), bladder bar
- Bionic charge bar (bionic dupes only)
- Attributes: icon grid of non-zero skill values (e.g. Digging 8, Construction 6)
- Traits: pill badges

Cards expand on click (future — not in scope for initial implementation).

---

### Tab 3 — Research

Two-column layout.

**Left column:**
- Active research card: current tech name + progress bar + % complete
- Unlocked technologies: icon grid (52×52px icons, emoji + name label, checkmark badge).
  Click an icon to expand a detail panel below the grid showing: tech name, unlock cycle,
  type, unlocked buildings list.

**Right column:**
- Printing Pod card (orange border when action pending):
  - Status: waiting / cooldown timer
  - Current offers: list of offer cards (name + traits/contents)
  - Next pod timer
  - Highlighted orange when overdue (AI obligation pending)

---

### Tab 4 — Perimeter

Two-column layout, left column spans both rows.

**Left column (full height):**
- Active perimeter card: goal name, bounds, blueprint, progress bar (% complete, buildings placed)
- Task board: Completed / Up Next / Blocked sections with task rows (checkmark state, dep note)
- Prerequisites missing: list of resource shortfalls vs. requirements

**Right column top:**
- Reward tracking card: tick reward, episode total, survival/progress/event layer breakdown
- Open events list (ongoing penalties with per-cycle cost)
- Recent events list (completed milestones with reward)

**Right column bottom:**
- Perimeter history: archived perimeters with goal, cycle range, result

---

### Tab 5 — Log

Two-column layout: main feed + stats sidebar.

**Main feed:**
- Filter pills: All / Actions / ACKs / Rewards / Events / Agent / Errors
- Scrollable timestamped log entries
- Each entry: timestamp | type badge | message
- Type badge colours:
  - `action` — blue
  - `ack ✓` — green
  - `ack ✗` — red
  - `reward` — purple
  - `agent` — grey
  - `error` — red
  - `event` — orange
- no-op entries rendered muted

**Stats sidebar (three cards):**
- Session Stats: ticks, actions sent, ACK success, ACK fail, no-ops, errors
- LLM Usage: calls, tokens in, tokens out, cost, avg latency
- Episode: phase, cycle/max, total reward, deaths

---

### Tab 6 — Config

Two-column layout.

**Left column — LLM Backend Profiles:**
- Named profile list: each entry shows name, endpoint host:port, model, active indicator (green dot)
- One profile marked active at a time
- "+ Add Profile" button
- Edit form below list (loads on profile click):
  - Profile Name (text)
  - Endpoint URL (text) — any OpenAI-compatible URL
  - Model (text)
  - API Key (optional, text)
  - Vision Pipeline toggle (enabled / disabled)
  - Set Active / Save / Delete buttons
  - Live connection badge (● Active · model · latency, or ● Error)
- Apply & Reconnect re-routes runner immediately, no restart needed

**Right column — single-config cards:**

*Game Connection (ONIBridge TCP):*
- Live reachability badge
- Host + Port fields
- Apply / Test / Disconnect buttons

*Runner Relay (WebSocket):*
- Live status badge
- Relay Port field
- Note: port change requires runner restart

*Runner Process:*
- Live status badge (running/stopped + pid + uptime)
- Restart / Stop buttons
- Note: restart picks up all config changes

---

## Colony Health Strip

Always visible, pinned between tab body and toolbar. Shows:

- `Colony Health` label
- Tick reward (green if positive, red if negative)
- Episode total reward
- Survival count (alive/total)
- Open obligations (orange pill per obligation, e.g. "⚠ print_pod_ready: 3 cycles unpicked")
- Phase · cycle/max (right-aligned, muted)

---

## Data Sources

All data comes from the runner relay WebSocket (live state pushes from ONIBridge). The dashboard
is read-only except for:

1. Config tab — LLM profile management, connection settings, runner start/stop/restart
2. Speed controls in toolbar (existing xdotool path)
3. Settings button in toolbar (existing xdotool path)

No new game actions are added by the dashboard redesign.

### New state fields required from ONIBridge (added in Phase 1):

| Field | Source |
|-------|--------|
| `duplicants[].skills` | StateSerializer — Klei.AI attributes |
| `duplicants[].traits` | StateSerializer — Klei.AI Traits component |
| `duplicants[].hunger` | StateSerializer — CalorieMonitor.Instance |
| `duplicants[].bladder` | StateSerializer — Db.Get().Amounts.Bladder |
| `duplicants[].type_data` | StateSerializer — RobotBatteryMonitor (bionic) |
| `research.unlocked` | StateSerializer — Research.Instance |
| `research.current_tech` | StateSerializer — Research.Instance |
| `research.current_progress` | StateSerializer — Research.Instance |

Perimeter data, reward tracking, and log entries are maintained client-side in the runner/dashboard
layer — not from ONIBridge directly.

---

## LLM Backend Config

Profiles are stored in a config file (JSON) on the dashboard server. The runner reads the active
profile at startup and accepts a hot-swap signal from the dashboard to reconnect without restart.

Config file location: `examples/dashboard/llm_profiles.json`

Schema per profile:
```json
{
  "id": "dgx-a",
  "name": "DGX-A",
  "endpoint_url": "http://10.0.0.69:8000/v1",
  "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
  "api_key": "",
  "vision_enabled": true
}
```

---

## Out of Scope

- Failover / fallback rules between profiles (future)
- Twitch overlay variant (future)
- Operator-triggered game actions from dashboard (by design — AI decides)
- Storage inventory icon grid on Colony tab (deferred to follow-up)
- Vision model serving on Linux desktop (tracked separately in model-catalog.md)
- NemoClaw / training pipeline config (separate spec)
