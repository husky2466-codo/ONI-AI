# Action Space — Design Spec

**Date:** 2026-04-04
**Status:** Approved for implementation

---

## Overview

This spec is the canonical reference for every action the AI agent can issue to the game.
It documents current actions, new additions, the implementation mechanism for each, and
a key architectural distinction that determines how each action is implemented in the mod.

---

## Harmony-Direct vs. UI-Automation

This distinction determines how each action is implemented in `ActionExecutor.cs`:

**Harmony-direct:** The game action has an underlying C# API that Harmony can call directly
from the mod, bypassing the UI entirely. This is the preferred approach — fast, reliable,
survives UI changes. All existing actions (`place_building`, `dig`, etc.) use this pattern.

**UI-automation (xdotool):** The action is only triggerable via UI interaction (clicking a
button, opening a menu). There is no backing C# API. In this case the mod sends a signal to
the Python runner, which uses `xdotool` on the Linux game host to simulate the click.
Currently used for: `set_speed` (SpeedControlScreen), `open_settings`.

The rule: if a human clicking a button calls a C# method under the hood, Harmony can call
that method directly. If the button IS the mechanism (pure UI), xdotool is required.

---

## Complete Action Reference

### Existing Actions (already implemented)

| Action | Mechanism | Parameters | Notes |
|--------|-----------|------------|-------|
| `place_building` | Harmony-direct | `building_id`, `cell_x`, `cell_y` | Calls `BuildingDef.TryPlace()` |
| `dig` | Harmony-direct | `cell_x`, `cell_y` | Creates dig chore directly |
| `cancel_dig` | Harmony-direct | `cell_x`, `cell_y` | Cancels queued dig chore |
| `set_priority` | Harmony-direct | `cell_x`, `cell_y`, `priority` (1–9) | Sets chore priority |
| `set_speed` | UI-automation | `speed` (0–3) | xdotool clicks SpeedControlScreen |
| `no_op` | — | — | Agent passes its turn |

---

### New Actions: Perimeter System (from Spatial Perimeter spec)

| Action | Mechanism | Parameters |
|--------|-----------|------------|
| `place_perimeter` | Harmony-direct | `x1`, `y1`, `x2`, `y2`, `goal` |
| `abandon_perimeter` | Harmony-direct | — |

Implementation: `PerimeterManager.Place()` / `PerimeterManager.Abandon()` — mod-side data
only, no game sim interaction. See Spatial Perimeter System spec for full detail.

---

### New Actions: Research

#### `assign_research`

**Mechanism: Harmony-direct** — Research assignment calls `Research.Instance` C# methods
internally when the player clicks a tech. Harmony calls the same methods directly.

**Parameters:** `{ "tech_id": "FarmingTech" }`

**C# implementation (`ActionExecutor.cs`):**

```csharp
case "assign_research":
{
    string techId = cmd.GetString("tech_id");
    if (string.IsNullOrEmpty(techId)) return "error_missing_tech_id";

    var db = Db.Get();
    if (db == null) return "error_db_unavailable";

    // Find the tech by ID in the tech database
    Tech tech = db.Techs.TryGet(techId);
    if (tech == null) return $"error_tech_not_found:{techId}";

    // Check tech is not already complete
    var techInstance = Research.Instance?.GetTechProgress(tech);
    if (techInstance == null) return "error_research_unavailable";
    if (techInstance.IsComplete()) return "error_already_complete";

    // Set as active research
    // Verify exact method name via decompile — likely SetActiveResearch or similar
    Research.Instance.SetActiveResearch(techInstance);
    return "ok";
}
```

**Decompile note:** Verify `Research.Instance.SetActiveResearch()` method name and signature.
Search for `activeResearch` setter in `Assembly-CSharp.dll`. The pattern is confirmed to
exist — ONI mods routinely automate research queues this way.

**Protocol (`protocol.py`):**
```python
def build_assign_research(tech_id: str) -> dict:
    return {"type": "action", "action": "assign_research", "tech_id": tech_id}
```

---

### New Actions: Building Control

#### `enable_building`

Toggles a building's enabled state (the on/off switch in the building UI). Used for power
management, automation override, and disabling machines that are starving for inputs.

**Mechanism: Harmony-direct** — `Operational` component has direct enable/disable API.

**Parameters:** `{ "cell_x": int, "cell_y": int, "enabled": bool }`

**C# implementation (`ActionExecutor.cs`):**

```csharp
case "enable_building":
{
    int cx = cmd.GetInt("cell_x"), cy = cmd.GetInt("cell_y");
    bool enable = cmd.GetBool("enabled");
    int cell = Grid.XYToCell(cx, cy);
    var go = Grid.Objects[cell, (int)ObjectLayer.Building];
    if (go == null) return "error_no_building";

    var op = go.GetComponent<Operational>();
    if (op == null) return "error_not_operational";

    // UserControlledToggle or BuildingEnabledButton handles the enable/disable flag
    // Verify exact mechanism via decompile — likely sets a flag on Operational or
    // a BuildingEnabled component
    var toggle = go.GetComponent<UserControlledToggle>();
    if (toggle != null)
    {
        toggle.Toggle(enable);
        return enable ? "enabled" : "disabled";
    }

    // Fallback: direct Operational flag if no toggle component
    op.SetFlag(Operational.Flag.Enabled, enable);  // verify flag name
    return enable ? "enabled" : "disabled";
}
```

**Decompile note:** Verify `UserControlledToggle.Toggle()` and `Operational.Flag.Enabled`.
Search for `BuildingEnabledButton` in Assembly-CSharp — this is the component the UI uses.

**Protocol:**
```python
def build_enable_building(cell_x: int, cell_y: int, enabled: bool) -> dict:
    return {"type": "action", "action": "enable_building",
            "cell_x": cell_x, "cell_y": cell_y, "enabled": enabled}
```

---

### New Actions: Printing Pod

#### `accept_print`

**Mechanism: UNCERTAIN — requires decompile verification before implementation.**

When the printing pod offers choices, the player selects one via the UI. Whether this
goes through a clean C# API or is purely UI-driven is unknown without decompiling
`Assembly-CSharp.dll`.

**If Harmony-direct is possible** (preferred):
```csharp
case "accept_print":
{
    int offerIndex = cmd.GetInt("offer_index");  // 0, 1, or 2
    // Call Immigration.Instance acceptance API
    // Verify: Immigration.Instance.AcceptImmigrant(int index) or similar
    return "ok";
}
```

**If UI-automation required:**
The mod returns a signal to the Python runner: `{ "type": "ui_required", "action": "accept_print", "offer_index": N }`. The runner uses `xdotool` to click the appropriate button in the immigration UI — same pattern as the settings button.

**Decompile checklist for this action:**
- [ ] `Immigration.Instance` — find accept/select method
- [ ] `ImmigrantPerks` — find offer indexing
- [ ] `CargoLander` or `MinionStartingStats` — understand offer types

---

## Updated `VALID_ACTIONS` (protocol.py)

```python
VALID_ACTIONS = {
    # Existing
    "place_building",
    "dig",
    "cancel_dig",
    "set_priority",
    "set_speed",
    "no_op",
    # Perimeter system
    "place_perimeter",
    "abandon_perimeter",
    # Research
    "assign_research",
    # Building control
    "enable_building",
    # Printing pod
    "accept_print",
}
```

---

## Actions Deliberately Not Included (Phase 1)

| Action | Reason deferred |
|--------|----------------|
| `assign_dupe_job(dupe_id, job, priority)` | Useful but not blocking Phase 1 — dupes use default job priorities initially |
| `assign_schedule(dupe_id, schedule)` | Deferred to Phase 2 — default schedule is survivable |
| `cancel_build(x, y)` | Low priority — agent can work around by not issuing conflicting builds |
| `set_building_filter(x, y, filter)` | Phase 2 — needed for advanced storage management |
| `open_settings` | UI-automation only, already exists via dashboard, not an AI action |

---

## Files Changed

| File | Change |
|------|--------|
| `mod/ONIBridge/src/ActionExecutor.cs` | Add `assign_research`, `enable_building`, `accept_print` handlers |
| `mod/ONIBridge/src/ActionCommand.cs` | Add `tech_id`, `enabled` fields |
| `src/agent/protocol.py` | Add new actions to `VALID_ACTIONS`, add builder functions |

---

## Decompile Checklist

- [ ] `Research.Instance.SetActiveResearch()` — exact method name for assign_research
- [ ] `UserControlledToggle.Toggle()` — building enable/disable mechanism
- [ ] `Operational.Flag.Enabled` — fallback flag name
- [ ] `Immigration.Instance` accept method — determines Harmony-direct vs. xdotool for accept_print
