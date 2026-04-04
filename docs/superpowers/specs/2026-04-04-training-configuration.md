# Training Configuration — Design Spec

**Date:** 2026-04-04
**Status:** Approved

---

## Overview

This spec documents the canonical training configuration for the ONI-AI agent: the game seed,
colony type decisions, episode lifecycle, and the reasoning behind each choice.

---

## Canonical Training Seed

```
v-sndst-c-1427943156-0-1a-j3et5
```

**Breakdown:**
- `sndst` — Sandstone starting asteroid (Terra equivalent). Temperate biome, standard resources,
  most well-documented starting asteroid in the community. Best choice for initial training.
- `c` — Cluster configuration (Spaced Out DLC)
- `1427943156` — Deterministic seed
- `1a` — DLC/layout variant
- `j3et5` — Additional cluster configuration hash

**Why a fixed seed matters:**
- Identical map layout every run: same geyser locations, biome boundaries, resource deposits
- Enables the agent to learn map-specific knowledge (e.g., "water geyser is NE of start")
- Reproducible failure analysis: when the agent dies at cycle 40, the map is identical to the
  previous run, isolating the agent's decisions as the variable
- Required for stable RL training: environment variance is a known enemy of sample efficiency

**When to introduce seed variety:** After the agent demonstrates consistent survival to cycle
50+ on the canonical seed. Introduce 3–5 alternate seeds to test generalization.

---

## Colony Type: Organic vs. Bionic

### Current configuration
3 organic dupes at start. Bionics available via printing pod (DLC active).

### Recommendation: Train on all-organic first

**Reasoning:**

| Factor | Organic | Bionic |
|--------|---------|--------|
| Community blueprints | Extensive | Limited |
| Wiki knowledge base | Comprehensive | Sparse |
| Survival constraint complexity | High | Low |
| Resource chain complexity | High (food → cook → eat → morale) | Low (power → charge) |
| Failure modes | Many, cascading | Few, linear |

Counterintuitively, **organic is the better initial training target** because:
1. The blueprint library (SPOM, food setups, morale buildings) is built for organic play
2. The wiki DB has far more data on organic survival buildings
3. The agent learns to reason about multi-variable constraint systems — harder, but the skill
   generalizes. An agent that masters organic survival understands the game's underlying logic.

**Bionic training (Phase 2):** Once organic survival to cycle 50 is demonstrated, run a
parallel bionic-only configuration. Power management is a single linear resource — the agent
will likely outperform organic quickly. The delta between organic and bionic performance tells
you exactly which survival skills the agent has internalized vs. which it was pattern-matching
from blueprints.

### Forcing homogeneous colony type

For training stability, avoid mixed organic/bionic colonies during Phase 1. Mixed colonies
require the agent to reason about two different survival trees simultaneously. Configure the
game or printing pod acceptance logic to enforce:

- **Phase 1:** Accept only organic dupe offers; accept care packages; skip bionic offers
- **Phase 2:** Accept only bionic dupe offers (separate training run)
- **Phase 3:** Mixed colony (advanced generalization test)

Implement this as a configurable policy in `runner.py`:

```python
COLONY_TYPE_POLICY = "organic_only"  # "organic_only" | "bionic_only" | "mixed"

def _should_accept_offer(offer: dict) -> bool:
    if offer["type"] == "care_package":
        return True  # always accept resource packages
    if offer["type"] == "duplicant":
        if COLONY_TYPE_POLICY == "organic_only":
            return offer["subtype"] == "organic"
        if COLONY_TYPE_POLICY == "bionic_only":
            return offer["subtype"] == "bionic"
    return True  # mixed: accept anything
```

The printing pod action decision is made by the AI, but the runner enforces this policy as a
hard constraint before sending `accept_print`.

---

## Episode Lifecycle

### Episode start
1. Game loaded from fresh save with canonical seed
2. ONIBridge mod initializes, TCP server opens on port 9999
3. Python runner connects, `SpatialLedger` initialized empty
4. `pending_actions` cleared
5. Episode counter incremented, start cycle logged

### Episode end conditions

| Condition | Trigger | Classification |
|-----------|---------|----------------|
| All dupes dead | `duplicants` list empty for 3 consecutive ticks | Loss |
| Cycle limit reached | `cycle >= MAX_CYCLE` (default: 100) | Neutral (for early training) |
| Colony goal achieved | Custom milestone (see below) | Win |
| Agent disconnect | TCP connection dropped without reconnect within 30s | Abort |

### Win conditions (phased)

Define win conditions that increase in difficulty as the agent improves:

| Phase | Win Condition | Description |
|-------|--------------|-------------|
| 0 | Stay alive for 3 cycles, no dupe deaths | Smoke test — validates the full loop works before tuning begins |
| 1 | Survive to cycle 10 | Dupes alive, oxygen positive, no starvation |
| 2 | Survive to cycle 25 with SPOM active | Self-sustaining oxygen |
| 3 | Survive to cycle 50 with 5+ dupes | Colony growth |
| 4 | Survive to cycle 100 | Long-run stability |

**Phase 0 rationale:** "Survive to cycle 10" is too binary as a smoke test — the agent either
trivially passes or dies at cycle 1–3 with no gradient to debug from. Phase 0 is achievable
with near-zero competent play (the starting conditions survive 3 cycles with almost no
actions) but immediately distinguishes a working loop from a broken one. Graduate to Phase 1
only after Phase 0 passes consistently across 5+ episodes.

### Episode reset

When an episode ends (any condition):
1. Log episode summary: start cycle, end cycle, end condition, final dupe count, final resources
2. Save episode data to `data/episodes/YYYYMMDD-HHMMSS-{seed}.jsonl`
3. Close TCP connection
4. Signal game to reload save (via `set_speed` to 0 then external process restart, or manual)
5. Wait for reconnect, begin next episode

**Game reload automation:** The game must be restarted or reloaded between episodes. This is
currently manual. Automation via `xdotool` (already used for settings) is feasible — send
key sequences to quit to main menu and reload the save. Design this in a separate spec.

**Priority note:** Game reload automation is the single biggest bottleneck for training
throughput. Manual reloads cap training at ~2–4 episodes per hour. Automated reloads could
reach 10–20+ per hour. This must be elevated to P1 once episode logging is confirmed
working — do not treat it as a polish item.

---

## Map-Specific Knowledge

The fixed seed means certain map facts are constants that can be baked into the system prompt
or a seed-specific knowledge file. Gather these facts from a human play-through of the seed:

```json
// data/seeds/v-sndst-c-1427943156.json (to be created after manual recon)
{
  "seed": "v-sndst-c-1427943156-0-1a-j3et5",
  "starting_biome": "Sandstone",
  "spawn_point": { "x": 120, "y": 200 },
  "known_geysers": [
    { "type": "cool_steam_vent", "approx_x": 145, "approx_y": 220, "notes": "NE of spawn" }
  ],
  "biome_warnings": [
    { "direction": "SW", "biome": "Slime", "note": "Do not dig SW before cycle 30" }
  ],
  "early_resources": [
    { "element": "IronOre", "approx_x": 108, "approx_y": 195 },
    { "element": "Algae",   "approx_x": 125, "approx_y": 185 }
  ]
}
```

This file is populated manually by a human playing the seed and noting key locations. It is
injected into the system prompt as static context, reducing the agent's need to discover
the map from scratch each episode.

---

## Episode Data Format

Each tick is logged to a JSONL file for training pipeline consumption:

```json
{
  "episode_id": "20260404-181200-1427943156",
  "cycle": 5,
  "tick": 1247,
  "state": { ... },
  "action": { "action": "dig", "cell_x": 115, "cell_y": 202 },
  "ack": { "success": true },
  "ledger_snapshot": { "active": null, "history": [] }
}
```

This is the raw training data. The reward function spec (separate document) defines how
these tuples are converted into `(state, action, reward)` triples for RL training.

---

## Configuration Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| Canonical seed | `v-sndst-c-1427943156-0-1a-j3et5` | Fixed for Phase 1 training |
| Colony type (Phase 1) | Organic only | Accept organic dupes + care packages |
| Colony type (Phase 2) | Bionic only | Separate training run, compare performance |
| Max dupes (Phase 1) | 5 | Constrain complexity during initial training |
| Episode cycle limit | 100 | Increase as agent improves |
| Tick rate | 1s real time (1x speed) | Matches current bridge timing |
| Win condition (Phase 1) | Survive cycle 10 | Achievable baseline to validate loop |

---

## Files Changed / Created

| File | Change |
|------|--------|
| `src/agent/runner.py` | Add `COLONY_TYPE_POLICY`, episode lifecycle management, episode JSONL logging |
| `data/seeds/` | New directory — seed-specific knowledge JSON files |
| `data/episodes/` | New directory — episode JSONL logs (gitignored) |
| `.gitignore` | Add `data/episodes/` |
