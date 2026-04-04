# Spatial Perimeter System — Design Spec

**Date:** 2026-04-04
**Status:** Approved for implementation
**Depends on:** Extended Game State Schema spec (storage + machine state required for prerequisite layer)

---

## Overview

The Spatial Perimeter System gives the AI a mechanism for declaring spatial intent on the map.
Instead of making reactive tick-by-tick decisions with no persistent plan, the AI places a named
bounding box (perimeter) around an area it intends to develop, attaches a goal (e.g.
`"oxygen_production"`), and receives a structured task board that tracks progress toward
completing the blueprint assigned to that goal.

This solves three problems simultaneously:
1. **Attention** — the AI works within one focused region rather than reasoning about the whole map
2. **Grounding** — a deterministic diff of blueprint vs. actual state gives the AI factual progress
3. **Persistent memory** — completed perimeters accumulate in a spatial ledger that survives across ticks

---

## Core Design Decisions

- **One active perimeter at a time** — the high-level planner sequences zones serially; dupes are
  a serial resource (3 dupes cannot meaningfully parallelize two construction zones)
- **Non-destructive overlay** — the perimeter exists only in mod-side data; it issues no game
  commands and dupes cannot interact with it
- **Goal-based blueprint selection** — the AI declares `goal="oxygen_production"`, the Python
  blueprint library selects the best-fit template based on goal + available space dimensions
- **Auto-complete** — when the blueprint diff reaches 100%, the perimeter archives itself
  automatically; no explicit `complete_perimeter` action required
- **Tiles window follows active perimeter** — the grid telemetry re-centers on the active
  perimeter bounds; reverts to default (building bounding box + 15 margin) when no perimeter is active

---

## Architecture

```
C# Mod (ONIBridge)              Python Agent
─────────────────────           ────────────────────────────────────────
PerimeterManager                SpatialLedger
  stores active perimeter         tracks active + archived perimeters
  renders overlay (visual)        owns focused_perimeter_id

StateSerializer                 BlueprintLibrary
  emits perimeter_state field     data/blueprints/*.json
  tiles window follows perimeter  goal → best-fit template selection

ActionExecutor                  TaskBoard
  handles place_perimeter           blueprint diff → ordered task list
  handles abandon_perimeter         dependency graph resolution

                                PrerequisiteResolver
                                  diffs build requirements vs. storage
                                  surfaces resource deficit tasks
```

---

## Part 1: C# Mod (ONIBridge)

### 1.1 New File: `PerimeterManager.cs`

A static manager class (same pattern as the existing static helpers) that stores the active
perimeter and handles overlay rendering.

```csharp
namespace ONIBridge
{
    public static class PerimeterManager
    {
        public static PerimeterData Active { get; private set; } = null;

        public static bool Place(int x1, int y1, int x2, int y2, string goal)
        {
            if (Active != null) return false;  // reject if already active
            Active = new PerimeterData
            {
                Id     = System.Guid.NewGuid().ToString("N").Substring(0, 8),
                X1 = x1, Y1 = y1, X2 = x2, Y2 = y2,
                Goal   = goal,
                Status = "active",
            };
            return true;
        }

        public static void Abandon()
        {
            Active = null;
        }

        // Called by StateSerializer each tick to produce the perimeter payload.
        // Completion detection (pct == 100) is handled on the Python side;
        // the mod just reports raw bounds and status. Python sends abandon_perimeter
        // when auto-complete triggers, which calls Abandon() here.
        public static object Serialize()
        {
            if (Active == null) return null;
            return new
            {
                id     = Active.Id,
                goal   = Active.Goal,
                bounds = new { x1 = Active.X1, y1 = Active.Y1, x2 = Active.X2, y2 = Active.Y2 },
                status = Active.Status,
            };
        }
    }

    public class PerimeterData
    {
        public string Id;
        public int X1, Y1, X2, Y2;
        public string Goal;
        public string Status;  // "active" | "abandoned"
    }
}
```

**Note on overlay rendering:** The perimeter boundary can be visualized using ONI's existing
`OverlayModes` system or by drawing debug lines via `GLDebug`. This is a polish item — the
functional behavior works without rendering. Implement rendering after the data flow is verified.

### 1.2 Changes to `ActionExecutor.cs`

Add two new action handlers in the `Execute` switch:

```csharp
case "place_perimeter":
{
    int x1 = cmd.GetInt("x1"), y1 = cmd.GetInt("y1");
    int x2 = cmd.GetInt("x2"), y2 = cmd.GetInt("y2");
    string goal = cmd.GetString("goal") ?? "unknown";
    bool ok = PerimeterManager.Place(x1, y1, x2, y2, goal);
    return ok ? "placed" : "rejected_perimeter_already_active";
}

case "abandon_perimeter":
{
    PerimeterManager.Abandon();
    return "abandoned";
}
```

### 1.3 Changes to `StateSerializer.cs`

**Add `perimeter` field to `Serialize()`:**

```csharp
public static object Serialize()
{
    return new
    {
        cycle      = TryGet("cycle",      GetCycle,      0),
        time       = TryGet("time",       GetTime,       0f),
        resources  = TryGet("resources",  GetResources,  (object)new {}),
        duplicants = TryGet("duplicants", GetDuplicants, new List<object>()),
        buildings  = TryGet("buildings",  GetBuildings,  new List<object>()),
        alerts     = TryGet("alerts",     GetAlerts,     new List<string>()),
        tiles      = TryGet("tiles",      GetTiles,      (object)new {}),
        perimeter  = TryGet("perimeter",  GetPerimeter,  (object)null),
    };
}

private static object GetPerimeter()
{
    return PerimeterManager.Serialize();
}
```

**Modify `GetTiles()` to follow active perimeter:**

Replace the window calculation block with:

```csharp
private static object GetTiles()
{
    int wx, wy, ex, ey;

    // If a perimeter is active, center the tile window on it
    var p = PerimeterManager.Active;
    if (p != null)
    {
        const int PAD = 5;
        wx = p.X1 - PAD;
        wy = p.Y1 - PAD;
        ex = p.X2 + PAD;
        ey = p.Y2 + PAD;
    }
    else
    {
        // Default: bounding box of all completed buildings + 15 tile margin
        // (existing logic unchanged — keep the current building bbox code here)
        // ... existing code ...
    }

    // Clamp to world bounds and cap at 64×64 — existing code unchanged below this point
    // ...
}
```

### 1.4 Protocol Changes

New action types (add to `ActionCommand.cs` and document in `protocol.py`):

```
place_perimeter:  { type, action, x1, y1, x2, y2, goal }
abandon_perimeter: { type, action }
```

The `perimeter` field in the state payload is `null` when no perimeter is active. Python treats
`null` as "no active perimeter."

---

## Part 2: Python Agent

### 2.1 New File: `src/agent/perimeter.py`

Contains `SpatialLedger`, `TaskBoard`, and `PrerequisiteResolver`.

#### SpatialLedger

```python
@dataclass
class LedgerEntry:
    id: str
    goal: str
    blueprint_id: str
    bounds: dict          # {x1, y1, x2, y2}
    status: str           # "complete" | "abandoned"
    cycle_started: int
    cycle_ended: int

class SpatialLedger:
    def __init__(self):
        self.active: Optional[ActivePerimeter] = None
        self.history: List[LedgerEntry] = []

    def on_state(self, state: dict) -> None:
        """Called each tick with the parsed state. Detects auto-complete."""
        p = state.get("perimeter")
        if p is None:
            self.active = None
            return
        if self.active is None:
            self.active = ActivePerimeter.from_state(p, state.get("cycle", 0))
            return
        self.active.update(state)
        if self.active.task_board and self.active.task_board.pct >= 100.0:
            self._archive("complete", state.get("cycle", 0))

    def _archive(self, status: str, cycle: int) -> None:
        if self.active:
            self.history.append(LedgerEntry(
                id=self.active.id, goal=self.active.goal,
                blueprint_id=self.active.blueprint_id,
                bounds=self.active.bounds, status=status,
                cycle_started=self.active.cycle_started, cycle_ended=cycle,
            ))
            self.active = None

    def format_context(self) -> str:
        """Returns a compact string for injection into the AI prompt."""
        lines = []
        if self.active:
            tb = self.active.task_board
            lines.append(f"ACTIVE PERIMETER: {self.active.goal} @ "
                         f"({self.active.bounds['x1']},{self.active.bounds['y1']})-"
                         f"({self.active.bounds['x2']},{self.active.bounds['y2']})")
            if tb:
                lines.append(f"  Progress: {tb.completed}/{tb.total} steps ({tb.pct:.0f}%)")
                for t in tb.next_tasks[:3]:
                    lines.append(f"  Next: {t}")
                for pr in tb.prerequisites[:3]:
                    lines.append(f"  Needs: {pr}")
        if self.history:
            lines.append(f"Completed zones: {len(self.history)} "
                         f"({', '.join(e.goal for e in self.history[-3:])})")
        return "\n".join(lines) if lines else "No active perimeter."
```

#### TaskBoard

The task board performs a deterministic diff of the blueprint template against actual buildings
reported in the current state:

```python
@dataclass
class TaskBoard:
    total: int
    completed: int
    pct: float
    next_tasks: List[str]       # ordered by dependency graph
    prerequisites: List[str]    # resource deficits blocking next tasks

    @classmethod
    def build(cls, blueprint: dict, state: dict) -> "TaskBoard":
        buildings_placed = {
            (b["type"], b["x"], b["y"]) for b in state.get("buildings", [])
        }
        steps = blueprint["buildings"]
        bounds = ...  # from active perimeter bounds
        origin_x, origin_y = bounds["x1"], bounds["y1"]

        completed = 0
        remaining = []
        for step in steps:
            abs_x = origin_x + step["rel_x"]
            abs_y = origin_y + step["rel_y"]
            if (step["type"], abs_x, abs_y) in buildings_placed:
                completed += 1
            else:
                remaining.append({**step, "abs_x": abs_x, "abs_y": abs_y})

        ordered = _topological_sort(remaining, blueprint["dependencies"])
        # Cycle resolution: ONI blueprints should never contain true dependency cycles
        # (A requires B, B requires A). If one is detected, break it by preferring the
        # node with fewer total dependents. If equal, select arbitrarily and log a warning.
        # A cycle in a blueprint JSON is a data error and should be fixed in the library.
        next_tasks = [
            f"Place {s['type']} at ({s['abs_x']},{s['abs_y']})"
            for s in ordered[:5]
        ]

        return cls(
            total=len(steps),
            completed=completed,
            pct=100.0 * completed / len(steps) if steps else 0.0,
            next_tasks=next_tasks,
            prerequisites=[],  # filled by PrerequisiteResolver
        )
```

#### PrerequisiteResolver

Diffs construction material requirements against storage inventory (requires Extended State
Schema storage field to be implemented):

```python
class PrerequisiteResolver:
    def resolve(self, next_steps: list, storage: list) -> List[str]:
        """
        Returns list of resource deficit strings.
        e.g. ["Need 200kg Refined Metal (have 0kg — mine iron ore + rock crusher)"]
        """
        available = self._aggregate_storage(storage)
        deficits = []
        for step in next_steps[:3]:
            for req in step.get("requires", []):
                have = available.get(req["element"], 0.0)
                need = req["mass_kg"]
                if have < need:
                    deficits.append(
                        f"Need {need:.0f}kg {req['element']} "
                        f"(have {have:.0f}kg)"
                    )
        return deficits

    def _aggregate_storage(self, storage: list) -> dict:
        totals = {}
        for container in storage:
            for item in container.get("contents", []):
                e = item["element"]
                totals[e] = totals.get(e, 0.0) + item["mass_kg"]
        return totals
```

### 2.2 New Directory: `data/blueprints/`

Each blueprint is a JSON file. The library is loaded at startup by `BlueprintLibrary`.

**Format (`data/blueprints/spom-v3.json`):**

```json
{
  "id": "spom-v3",
  "name": "Standard SPOM v3 (Self-Powered Oxygen Module)",
  "goals": ["oxygen_production"],
  "min_size": { "w": 8, "h": 10 },
  "buildings": [
    {
      "type": "LiquidPump",
      "rel_x": 0, "rel_y": 0,
      "requires": []
    },
    {
      "type": "Electrolyzer",
      "rel_x": 2, "rel_y": 2,
      "requires": [{ "element": "Water", "mass_kg": 50 }]
    },
    {
      "type": "HydrogenGenerator",
      "rel_x": 4, "rel_y": 4,
      "requires": []
    }
  ],
  "dependencies": {
    "Electrolyzer":       ["LiquidPump"],
    "HydrogenGenerator":  ["Electrolyzer"],
    "GasPump":            ["Electrolyzer"]
  },
  "dig_required": [
    { "rel_x": 1, "rel_y": 1 },
    { "rel_x": 2, "rel_y": 2 }
  ]
}
```

**Goal → Blueprint selection (`BlueprintLibrary.select`):**

```python
class BlueprintLibrary:
    def select(self, goal: str, available_w: int, available_h: int) -> Optional[dict]:
        candidates = [
            bp for bp in self._blueprints.values()
            if goal in bp["goals"]
            and bp["min_size"]["w"] <= available_w
            and bp["min_size"]["h"] <= available_h
        ]
        # Pick smallest blueprint that fits (prefer minimal resource footprint early)
        return min(candidates, key=lambda b: b["min_size"]["w"] * b["min_size"]["h"],
                   default=None)
```

### 2.3 Changes to `protocol.py`

Add new action types to `VALID_ACTIONS` and add builder functions:

```python
VALID_ACTIONS = {
    # ... existing ...
    "place_perimeter",
    "abandon_perimeter",
}

def build_place_perimeter(x1: int, y1: int, x2: int, y2: int, goal: str) -> dict:
    return {"type": "action", "action": "place_perimeter",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "goal": goal}

def build_abandon_perimeter() -> dict:
    return {"type": "action", "action": "abandon_perimeter"}
```

**Python-side overlap validation** (in `runner.py` before sending):

```python
def _validate_perimeter(self, x1, y1, x2, y2) -> bool:
    """Reject if a perimeter is already active."""
    return self._ledger.active is None
```

### 2.4 Changes to `runner.py`

- Instantiate `SpatialLedger` at startup
- Call `ledger.on_state(state)` each tick after parsing
- When `on_state` detects auto-complete (pct >= 100), send `abandon_perimeter` action to mod
  to clean up the mod-side `PerimeterManager`
- Pass ledger to `GeminiAgent` so it can include context in prompt

### 2.5 Changes to `llm.py`

Add ledger context to `_format_state()`:

```python
def _format_state(self, state: dict) -> str:
    # ... existing sections ...
    prompt += f"\nSpatial Ledger:\n{self._ledger.format_context()}\n"
    return prompt
```

---

## Part 3: Pending Actions Tracking

**This is a P0 fix required regardless of the perimeter system.**

The agent currently re-issues the same dig/build commands because it has no memory of what it
already ordered. Fix in `runner.py`:

```python
class Runner:
    def __init__(self):
        self._pending_actions: List[dict] = []  # actions sent, not yet confirmed
        # ...

    async def _send_action(self, action: dict):
        self._pending_actions.append(action)
        await self._client.send(action)

    def _on_ack(self, ack: dict):
        # Remove from pending on success or permanent failure
        if ack.get("success") or ack.get("error") == "already_queued":
            self._pending_actions = [
                a for a in self._pending_actions
                if not self._matches(a, ack)
            ]

    def _format_pending(self) -> str:
        if not self._pending_actions:
            return ""
        lines = ["Pending (already issued, do not re-issue):"]
        for a in self._pending_actions[-10:]:
            lines.append(f"  {a['action']} @ ({a.get('x1') or a.get('cell_x','?')},...)")
        return "\n".join(lines)
```

Include `_format_pending()` output in the state prompt via `_format_state()`.

---

## State Payload — `perimeter` Field

When active:

```json
{
  "perimeter": {
    "id": "a3f9c1e2",
    "goal": "oxygen_production",
    "bounds": { "x1": 32, "y1": 18, "x2": 40, "y2": 28 },
    "status": "active"
  }
}
```

When no perimeter: `"perimeter": null`

The task board (progress, next_tasks, prerequisites) is computed entirely on the Python side
from the blueprint diff and storage state. It is not stored in the mod.

---

## Prompt Context Example

When the AI receives a state tick with an active perimeter, the state prompt includes:

```
ACTIVE PERIMETER: oxygen_production @ (32,18)-(40,28)
  Progress: 7/15 steps (46%)
  Next: Place GasPump at (35,22)
  Next: Place GasPump at (35,24)
  Next: Place Electrolyzer at (36,23)
  Needs: 200kg Refined Metal (have 0kg)

Completed zones: 1 (base_camp)
```

---

## Files Changed

| File | Change |
|------|--------|
| `mod/ONIBridge/src/PerimeterManager.cs` | New file |
| `mod/ONIBridge/src/StateSerializer.cs` | Add `perimeter` field, modify `GetTiles` window |
| `mod/ONIBridge/src/ActionExecutor.cs` | Add `place_perimeter`, `abandon_perimeter` handlers |
| `mod/ONIBridge/src/ActionCommand.cs` | Add x1, y1, x2, y2, goal fields |
| `src/agent/perimeter.py` | New file — SpatialLedger, TaskBoard, PrerequisiteResolver |
| `src/agent/protocol.py` | Add new action types + builder functions |
| `src/agent/runner.py` | Pending actions tracking, ledger integration, auto-complete handler |
| `src/agent/llm.py` | Ledger context in `_format_state()` |
| `data/blueprints/` | New directory — blueprint JSON library |

---

## Out of Scope

- Multiple concurrent perimeters (future phase after single-perimeter is proven)
- Visual overlay rendering in-game (polish item, functional without it)
- Automated blueprint generation (AI uses library; custom blueprints are future work)
- Inter-perimeter dependency graph (future phase)
