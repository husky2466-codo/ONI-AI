# Pipeline Inspector — Design Spec

**Date:** 2026-04-04
**Status:** Approved for implementation

---

## Goal

Add a live, per-tick visibility layer to the ONI-AI dashboard so the operator can see exactly
what is happening inside the inference pipeline at every stage — without editing code or reading
log files.

---

## Problem

The current pipeline runs invisibly. The dashboard Log tab shows only the final action per tick.
When something goes wrong (LLM hallucinates a coordinate, validator blocks an action, game
rejects a command), there is no way to see *which stage* failed or *why* without SSHing into the
machine and tailing `runner.log`. This makes debugging require code knowledge and terminal access.

---

## Architecture

### New message type: `pipeline`

`runner.py` captures a snapshot of every pipeline stage for each tick and broadcasts it to
connected dashboard clients as a new WebSocket message type alongside the existing `state` and
`ack` types.

```
runner.py  →  PipelineSnapshot (per tick)
           →  broadcasts {"type": "pipeline", "tick": N, "cycle": N, "stages": [...]}
           →  server.py stores last 50 snapshots
           →  browser renders in new "Pipeline" tab
```

### PipelineSnapshot structure

Each snapshot contains one entry per stage:

```json
{
  "type": "pipeline",
  "tick": 42,
  "cycle": 3,
  "elapsed_ms": 8420,
  "stages": [
    {
      "name": "state_in",
      "label": "State Received",
      "data": {
        "cycle": 3,
        "dupes": 3,
        "o2_kg": 4.2,
        "alerts": 0,
        "tile_window": "64x64 @ (100,190)"
      }
    },
    {
      "name": "prompt",
      "label": "Prompt Formatted",
      "data": {
        "chars": 2840,
        "tokens_est": 710,
        "preview": "Cycle: 3\n\nResources:\n  oxygen_kg: 4.20..."
      }
    },
    {
      "name": "llm_call",
      "label": "LLM Response",
      "data": {
        "model": "gemini-2.5-flash",
        "elapsed_ms": 8120,
        "raw_response": "I will dig at (114,214) to expand the base...\n```json\n{\"action\": \"dig\", ...}```",
        "extracted_json": {"action": "dig", "cell_x": 114, "cell_y": 214}
      }
    },
    {
      "name": "validation",
      "label": "Validation",
      "data": {
        "input_action": {"action": "dig", "cell_x": 114, "cell_y": 214},
        "result": "blocked",
        "reason": "cell (114,214) is not solid — already open",
        "output_action": {"action": "no_op"}
      }
    },
    {
      "name": "sent",
      "label": "Sent to Game",
      "data": {
        "action": {"action": "no_op"}
      }
    },
    {
      "name": "ack",
      "label": "Game ACK",
      "data": {
        "action": "no_op",
        "success": true,
        "error": null
      }
    }
  ]
}
```

---

## Components

### 1. `src/agent/runner.py` — PipelineCapture

Add a `PipelineCapture` dataclass that collects stage data during the tick loop and emits
the final snapshot at the end of each tick.

Key stages captured:
- `state_in` — summary of incoming state (cycle, dupe count, O2, alerts, tile window size)
- `prompt` — character count, estimated tokens, first 500 chars of formatted prompt
- `llm_call` — model name, elapsed ms, first 800 chars of raw LLM response, extracted JSON
- `validation` — input action, result (passed/blocked/deduped), reason, output action
- `sent` — final action dict sent to game
- `ack` — game ack (action, success, error)

The LLM elapsed time is captured by timing the `run_in_executor` call.
The prompt text is captured by having `llm.py` return it alongside the action (see below).

### 2. `src/agent/llm.py` — Return prompt alongside action

`GeminiAgent.decide()` currently returns only a `dict` (the action). It must also return the
formatted prompt text and raw LLM response so `runner.py` can include them in the snapshot.

Change signature:
```python
# Before
def decide(...) -> dict:

# After
def decide(...) -> tuple[dict, str, str]:
    # returns (action_dict, prompt_text, raw_llm_response)
```

`runner.py` unpacks the tuple and stores prompt/response in the `PipelineCapture`.

### 3. `examples/dashboard/server.py` — Pipeline snapshot storage

Add `pipeline_snapshots: list[dict] = []` (capped at 50 entries).
On new browser connect, send the last 10 snapshots so the tab is pre-populated.
On each `pipeline` message from relay, append to list and broadcast to browsers.

### 4. `examples/dashboard/index.html` — Pipeline tab

New tab: **"Pipeline"** between Log and Config.

Layout: two-panel.
- **Left panel:** scrolling feed of ticks (most recent at top). Each tick is a collapsed row
  showing: tick number, cycle, elapsed ms, final action, validation result badge.
  Click a row to expand it and see all stages.
- **Right panel:** expanded detail of the selected tick. Each stage shown as a card with its
  label, status badge, and formatted data. Prompt and LLM response shown in scrollable
  `<pre>` blocks. Auto-selects the latest tick when auto-scroll is on.

Controls:
- **Auto-scroll toggle** — when on, always selects and shows the latest tick
- **Pause** — freeze the feed (don't add new ticks) for inspection
- **Clear** — wipe the feed

Stage status badges:
- `passed` — green
- `blocked` — orange (validator blocked, different action sent)
- `deduped` — grey (same as last tick, suppressed)
- `failed` — red (LLM error, parse error, game rejection)
- `ok` — green (ack success)

---

## Validation result classification

In `runner.py`, after all validation steps, classify the outcome:

| Condition | Result label |
|-----------|-------------|
| candidate sent as-is | `passed` |
| place_perimeter hard-blocked | `blocked` (reason: perimeter already active) |
| dig coordinate not solid | `blocked` (reason: cell not solid) |
| dedup suppressed | `deduped` |
| LLM parse failed → no_op | `failed` |
| manual action override | `manual` |

---

## Token estimation

Prompt token count estimated as `len(prompt_text) / 4` (rough approximation). No external
tokenizer library required.

---

## Snapshot cap

`server.py` keeps the last 50 pipeline snapshots in memory. The browser receives the last 10
on connect. There is no persistence to disk — snapshots are session-only and lost on server
restart.

---

## Files Changed

| File | Change |
|------|--------|
| `src/agent/llm.py` | `decide()` returns `(action, prompt_text, raw_response)` tuple |
| `src/agent/runner.py` | Add `PipelineCapture`, collect stages per tick, broadcast `pipeline` message |
| `examples/dashboard/server.py` | Store last 50 pipeline snapshots, send last 10 on connect, broadcast to browsers |
| `examples/dashboard/index.html` | Add Pipeline tab with two-panel layout, tick feed, stage cards |

---

## What this enables

- See the exact prompt the LLM received each tick
- See the raw LLM response text (before JSON extraction)
- See which validation step blocked an action and why
- See how long the LLM call took
- Identify whether bad behavior is a prompt problem, an LLM problem, or a validation problem
- Do all of this from the browser — no terminal, no log files
