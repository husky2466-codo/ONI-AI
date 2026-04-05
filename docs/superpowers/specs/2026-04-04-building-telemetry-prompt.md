# Building Telemetry Prompt Fix — Design Spec

**Date:** 2026-04-04
**Status:** Approved for implementation

---

## Problem

The ONIBridge mod correctly serializes all completed buildings — including game-spawned starting
structures (`Telepad`, `RationBox`, etc.) — and sends them in the `buildings` list every tick.
However, the LLM prompt in `llm.py` has two gaps that cause the agent to reason incorrectly:

1. **Unknown type IDs:** The system prompt's building reference table lists only player-buildable
   structures. When the LLM sees `Telepad @ (130,200) [OK]` in the state, it has no knowledge of
   what a `Telepad` is, so it ignores it or hallucinates its function.

2. **False survival warnings:** The survival checklist in `_format_state()` flags
   `"MicrobeMusher (no food production!)"` even when a `RationBox` is present, because it only
   checks for `MicrobeMusher` in `built_types`. This causes the agent to prioritize building a
   Microbe Musher when food is already available via the ration box.

No C# changes are required — the data pipeline is correct. The fix is entirely in `llm.py`.

---

## Architecture

Three changes to `src/agent/llm.py`:

1. **System prompt:** Add a "Starting buildings" reference table so the LLM knows the type IDs,
   names, and functions of game-spawned structures present on every new map.

2. **Survival checklist:** Update the food-source check to recognize `RationBox` as an existing
   food supply. Warn about `MicrobeMusher` as a long-term need rather than an immediate crisis
   when a ration box is present.

3. **Buildings prompt section:** Tag spawned buildings with `[SPAWNED]` so the LLM can
   visually distinguish pre-existing assets from player-built ones.

---

## Changes

### 1. System prompt — Starting buildings table

Add after the existing `### Mid-game oxygen` section, before `## Survival rules`:

```
## Starting buildings (pre-spawned — always present on a new map)

These buildings exist at game start. Do NOT try to place them — they are already built.

| type (in state) | In-game name    | Function |
|-----------------|-----------------|----------|
| Telepad         | Printing Pod    | Delivers duplicants or care packages every 3 cycles. Check printing_pod.cycles_until_next in state. |
| RationBox       | Ration Box      | Stores up to 20 kg food. Starting food supply lives here. Does NOT produce food — needs a cooker for long-term. |
| IceCooledFan    | Ice-Cooled Fan  | Cools nearby area using ice. Stops when ice runs out. |
| StorageLocker   | Storage Locker  | General-purpose raw material storage. |
```

### 2. Survival checklist — food source logic

Current code (lines ~225-232 of `_format_state()`):

```python
if "MicrobeMusher" not in built_types:
    survival.append("MicrobeMusher (no food production!)")
```

Replace with:

```python
_FOOD_PRODUCERS = {"MicrobeMusher", "GasRangeComplete", "ElectricGrillComplete"}
if not (built_types & _FOOD_PRODUCERS):
    if "RationBox" in built_types:
        survival.append("MicrobeMusher (ration box has starting food, but build a cooker soon)")
    else:
        survival.append("MicrobeMusher (no food source!)")
```

### 3. Buildings prompt section — spawned tag

Current code renders all buildings the same way. Add a `[SPAWNED]` tag for known spawned types:

```python
_SPAWNED_TYPES = {"Telepad", "RationBox", "IceCooledFan", "StorageLocker"}

lines.append(f"Buildings on map: {len(buildings)}")
for b in buildings[:20]:
    op = "OK" if b.get("operational") else "OFFLINE"
    tag = " [SPAWNED]" if b.get("type") in _SPAWNED_TYPES else ""
    lines.append(f"  {b.get('type','?')}{tag} @ ({b.get('x','?')},{b.get('y','?')}) [{op}]")
if len(buildings) > 20:
    lines.append(f"  ...and {len(buildings) - 20} more")
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/agent/llm.py` | Add starting buildings table to `SYSTEM_PROMPT`; update food-source check in `_format_state()`; add `[SPAWNED]` tag to buildings prompt section |

---

## Testing

- Unit test: `_format_state()` with a state containing only `RationBox` (no `MicrobeMusher`) should
  produce a warning about building a cooker soon, not "no food source!"
- Unit test: `_format_state()` with a state containing no food buildings at all should produce
  the urgent "no food source!" warning
- Unit test: `RationBox` in built_types should NOT appear in the survival warning list
- Manual: run agent with game in early cycle, check Pipeline Inspector prompt stage — verify
  `Telepad [SPAWNED]` and `RationBox [SPAWNED]` appear, verify no false food warning

---

## What This Enables

- LLM correctly understands what `Telepad` is and monitors `printing_pod.cycles_until_next`
- LLM correctly understands `RationBox` is a food storage, not a food producer
- LLM stops urgently prioritizing MicrobeMusher when starting food supply exists
- Spawned buildings are visually distinct in the prompt so the LLM doesn't try to re-build them
