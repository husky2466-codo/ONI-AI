# First Live Agent Session — 2026-04-04

## Summary

First successful end-to-end run of the ONI-AI agent controlling a live game.

## Session Log (from Unity Player.log)

```
[17:56:54] ONIBridge mod loaded, listening on port 9999
[18:12:09] AI agent connected
[18:12:11] Dig queued at (115,202)
[18:12:21] Placed Bed at (116,201)
[18:12:43] Dig queued at (113,201)
[18:12:45] Dig queued at (113,202)
[18:12:53] Dig queued at (113,201) [repeat — dedup not firing across reconnect]
[18:13:16] AI agent disconnected (~4 min session)
```

## Observations

- Agent successfully issued real game commands: dig orders and building placement (Bed)
- **Spatial reasoning issue**: Agent lost track of where to place commands after a few actions.
  Likely cause: tile window coordinates in state vs. game coordinates not being interpreted
  correctly by Gemini — the tile window summary gives element counts but not a clear spatial
  map the LLM can reason about.
- **Duplicate dig at (113,201)**: sent twice, suggesting dedup resets on reconnect don't
  carry over correctly or the agent re-decided the same action on reconnect.
- **Stress values > 1.0** in state (observed 1.538, 1.522, 1.490) — StressMonitor.stress.value
  is not clamped to 0–1 in the serializer. May need `Math.Clamp`.
- **Food kcal = 16,000,000** — Edible.Calories appears to return raw grams (or a different
  unit), not kcal. Needs investigation.
- Session ended cleanly after ~4 minutes when runner was stopped.

## What Worked

- TCP bridge stable throughout session
- Gemini correctly parsed state JSON and returned valid action JSON
- Place building and dig actions executed in-game as expected
- Dashboard connected and showed live state

## Known Issues to Address in Next Session

1. Spatial reasoning — agent needs better coordinate context in prompt
2. Stress units — clamp or verify StressMonitor value range
3. Food kcal units — verify Edible.Calories unit (grams vs kcal)
4. Dedup across reconnects
