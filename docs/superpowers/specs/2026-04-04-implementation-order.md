# Implementation Order — Spec Dependencies

**Date:** 2026-04-04

The four specs in this directory have implicit dependencies. Implement in this order:

```
1. Spatial Perimeter System       (independent — can start immediately)
        ↓
2. Extended Game State Schema     (perimeter spec references storage for prerequisite resolver)
        ↓
3. Reward Function                (needs printing pod + dupe stats from state schema)
        ↓
4. Training Config plumbing       (needs reward calculator + episode logging from reward spec)
```

## Why this order

**Perimeter system is independent.** It adds new action types and a mod-side data structure.
Nothing else depends on it being done first, but it doesn't depend on anything either.
Start here to get the highest-value feature unblocked.

**Extended state schema before reward function.** The reward function's event detector
watches for printing pod state changes (`status: waiting_for_decision`) and dupe deaths
detected via the duplicants list. Both require the extended state schema to be accurate.
The prerequisite resolver in the perimeter task board requires the storage inventory field.

**Reward function before training config plumbing.** The episode JSONL log needs to record
per-tick reward alongside state and action. Can't finalize the log schema without the
reward calculator being defined.

**Training config plumbing last.** This is the glue layer — episode lifecycle management,
JSONL logging, game reload automation. It depends on everything else being stable.

## P0 items that cut across all specs

These should be done before or in parallel with Step 1:

- **Pending actions tracking** (`runner.py`) — Python-only, no C# changes, fixes the
  re-ordering bug that affects every live session right now
- **Stress value clamp** (`StateSerializer.cs`) — one-line fix, known bug from session 1
- **Food kcal unit fix** (`StateSerializer.cs`) — known bug from session 1

## Human play recording (future)

Existing save files are **snapshots, not trajectories** — they show a finished state but
not the sequence of decisions that built it. They are valuable as:
- Curriculum learning starting states (load mid-game save, train from there)
- Blueprint reference material for the blueprint library
- Validation benchmarks ("can the AI match this save by cycle X?")

For inverse RL (learning reward function from expert play), the user must play a fresh
session with recording enabled. The dashboard should add a "record human play" toggle
that routes manual actions through the same episode JSONL logger the AI uses.
This is deferred to a future spec.
