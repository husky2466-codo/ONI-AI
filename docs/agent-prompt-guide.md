# ONI-AI Agent Prompt Guide

## Overview

The Gemini agent (`src/agent/llm.py`) is guided by two things:
1. A **system prompt** (`SYSTEM_PROMPT`) — static instructions loaded once, defines the agent's knowledge and behavior
2. A **state prompt** — generated each tick by `_format_state()`, describes the current game state

This document explains what's in each, why it's structured that way, and how to improve it.

---

## System Prompt Design

The system prompt lives in `src/agent/llm.py` as `SYSTEM_PROMPT`. It covers:

### 1. Map & Coordinate Context

```
- 2D grid map. Duplicants start near center (~x=120, y=200 on a standard Sandstone map).
- Y increases upward. Tiles must be DUG before buildings can be placed in them.
```

**Why:** Gemini has no inherent knowledge of ONI's coordinate system. Without this, it places buildings in wrong locations or tries to place in solid tiles.

**Known issue:** Even with this, the agent loses spatial reasoning after a few actions. The tile window gives element counts but not a navigable map the LLM can use to plan a sequence of dig+place actions. Needs improvement — see session log.

### 2. Building Reference Table

A table of every building the agent can place in early game, with exact `building_id` values, power draw, inputs/outputs, and size. This is the single most important section — without it Gemini either invents wrong building IDs or doesn't know what to build next.

**Priority order:** Outhouse → Bed → ManualGenerator → Wire → Battery → OxygenDiffuser → MicrobeMusher → ResearchCenter

### 3. Survival Rules

Eight numbered rules that encode the most common early-game failure modes:
- Always have 1 Outhouse per 3 dupes
- Always have 1 Cot per dupe
- Power chain must be wired explicitly
- Don't place a building type already present nearby
- Issue one action per tick — no spamming

**Why rule 8 (one action per tick):** The TCP bridge fires at 1s intervals. If the agent tries to place 5 buildings in one response, only 1 goes through. The rest are lost. The agent must pace itself.

### 4. ONI Survival Knowledge

Embedded game knowledge the agent needs but can't get from state:
- Oxygen math (100g/dupe/second, OxygenDiffuser capacity)
- Food math (1000 kcal/dupe/day, Mush Bar recipe)
- Power chain math (total early draw ~420W)
- Gas physics (CO2 sinks, H2 rises)
- Stress mechanics (thresholds and causes)
- Research priority order
- Biome safety (don't dig slime biome)

**Why embed this instead of relying on wiki tool calling?** These are the facts needed on every single tick. Wiki tool calls add latency and token cost. The system prompt encodes the critical fast-path knowledge; wiki calls are for edge cases and look-up of specific building stats the agent doesn't already know.

### 5. Response Format

```
Output ONLY a single JSON object — no explanation, no markdown, no code fences
```

**Why:** Gemini naturally wants to explain its reasoning. This instruction, combined with `_parse_action()` stripping markdown fences, ensures the response is always parseable JSON.

---

## State Prompt Format

Generated each tick by `_format_state()`. Example output:

```
Cycle: 3

Resources:
  oxygen_kg:  0.42
  water_kg:   27.80
  food_kcal:  3200
  power_kw:   0.00
  co2_kg:     0.012

Duplicants (3):
  Lindsay @ (132,203) stress=12% hp=100 task=Dig
  Ruby @ (133,203) stress=8% hp=100 task=BuildFetch
  Otto @ (132,203) stress=9% hp=100 task=Idle

Buildings (10):
  Tile @ (127,202) [ok]
  ...
  Headquarters @ (124,203) [ok]
  RationBox @ (121,203) [ok]

MISSING survival buildings: Outhouse, Bed, OxygenDiffuser, ManualGenerator

Tile window: x=109 y=187 w=64 h=64 (4096 tiles) | top elements: Sandstone(2841), Vacuum(890), Dirt(210), Abyssalite(88), Oxygen(42), Water(25)
```

### Key sections

**MISSING survival buildings** — computed by diffing the buildings list against the required survival set. This is the agent's primary directive on every tick. If Outhouse is missing, it should dig and place one before doing anything else.

**Tile window summary** — top 6 elements by count in the 64×64 window around the base. Tells the agent what materials are available to dig. Does NOT give the agent a spatial map it can navigate — this is a known limitation.

---

## Wiki Tool Calling

Before deciding each action, Gemini can call `search_wiki(query)` up to 2 times. The wiki DB (`data/wiki.db`) is built by `scripts/build_wiki_db.py` from the ONI wiki.

**When Gemini uses it:** When it encounters an unfamiliar building or element in state. For example, if it sees `Electrolyzer` in the buildings list and doesn't know what inputs it needs.

**Cost:** Each wiki call is an additional Gemini API round-trip (~50-100ms latency, ~500 tokens). The 2-call limit prevents runaway tool use.

---

## Known Prompt Weaknesses (from first live session)

### 1. Spatial reasoning degrades quickly

The agent knows approximate starting coordinates (`~x=120, y=200`) but once it digs a few tiles it loses track of where free space is. The tile window summary shows *what* elements exist but not *where* specific open tiles are.

**Proposed fix:** Include a small ASCII map or a list of the nearest vacuum/open cells from the tile window data, so the agent can pick adjacent dig targets rather than re-issuing the same coordinates.

### 2. No memory of queued actions

The agent doesn't know which digs or builds it already ordered. If a dupe is mid-dig at (113,201), the agent may re-issue the same dig next tick. The buildings list only shows *completed* buildings, not in-progress ones.

**Proposed fix:** Track queued actions in runner.py and include them in the state prompt as "pending actions."

### 3. Stress and food unit bugs

- Stress reads >1.0 (should be 0.0–1.0) — `StressMonitor.stress.value` not clamped in C#
- Food kcal reads ~16,000,000 — `Edible.Calories` unit mismatch in C#

The system prompt works around food with `food_kcal values above 1,000,000 are save artifacts — ignore them`. Stress is displayed as a percentage (`stress*100`) which makes values like 153% visible — the agent interprets this as high stress (correct behavior, wrong number).

---

## Improving the Agent

When modifying the system prompt:
1. Edit `SYSTEM_PROMPT` in `src/agent/llm.py`
2. No C# rebuild needed — prompt changes take effect on next runner start
3. Test with a live session and check runner logs for `-> AI action:` lines

When modifying the state format:
1. Edit `_format_state()` in `src/agent/llm.py`
2. Run `pytest tests/agent/test_llm_wiki.py -v` to verify tile summary tests still pass
