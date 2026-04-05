# Spatial Perimeter System v2 — Design Spec

**Date:** 2026-04-05
**Status:** Approved for implementation
**Supersedes:** `docs/superpowers/specs/2026-04-04-spatial-perimeter-system.md`
**Depends on:** Extended Game State Schema spec (storage field required for PrerequisiteResolver)

---

## Overview

The Spatial Perimeter System v2 upgrades the original single-zone model to support multiple
concurrent build zones, each with an independent priority, blueprint, and task board.

The agent declares bounding boxes (zones) on the map with a goal and priority. The Python layer
validates placement, selects a blueprint, and tracks progress. The C# mod stores all active zones,
keeps them sorted by priority, and points the tile window at the highest-priority (focused) zone.

**What changed from v1:**
- `active: ActivePerimeter | None` → `zones: list[ActiveZone]` (max 5, sorted priority desc)
- `place_perimeter` gains `priority` (1–9) and `id` (Python-generated) params
- `abandon_perimeter` gains `id` param to target a specific zone
- Placement validation moved entirely to Python (blueprint match, overlap, padding, cap)
- Coordinate convention locked to exclusive bounds; 1-tile padding enforced on all blueprints
- `base-camp-v1` min_size bumped to 12×8 to satisfy padding rule
- State payload field renamed from `perimeter` (object|null) to `zones` (array)

---

## Core Design Decisions

- **Max 5 concurrent zones** — enforced Python-side and mod-side; prevents prompt bloat
- **Priority 1–9** — matches ONI errand priority convention; 9 = highest; tile window follows highest-priority zone
- **Lifecycle: active → complete | abandoned** — no pause state; zones only move forward
- **Fail-fast placement** — zone rejected before reaching the mod if blueprint match fails, bounds too small, overlap detected, or cap reached; rejection reason injected into next tick's prompt
- **Exclusive bounds** — `x2` and `y2` are exclusive; interior cells are `x1..x2-1`, `y1..y2-1`; width = `x2-x1`, height = `y2-y1`
- **1-tile padding** — no blueprint building or dig cell may touch the zone boundary; `rel_x <= (x2-x1)-2`, `rel_y <= (y2-y1)-2`
- **Python-generated zone ID** — UUID4 short hex generated before the action is sent; Python knows the ID immediately without a round-trip
- **Visual overlay** — deferred; functional without rendering

---

## Architecture

```
C# Mod (ONIBridge)                  Python Agent
─────────────────────               ──────────────────────────────────────────
PerimeterManager                    SpatialLedger
  List<PerimeterData> sorted          zones: list[ActiveZone] sorted priority desc
  by priority desc                    validate_place() — all checks before mod
  Focused = zones[0]                  on_state() — sync from mod, update task boards
                                      format_context() — prompt injection
StateSerializer
  emits zones[] array               BlueprintLibrary
  tile window follows Focused         data/blueprints/*.json
                                      select(goal, w, h) → best-fit template
ActionExecutor
  place_perimeter(id,x1,y1,x2,y2,  TaskBoard
    goal,priority)                    blueprint diff → ordered task list
  abandon_perimeter(id)               dependency graph resolution

                                    PrerequisiteResolver
                                      diffs build requirements vs. storage
```

---

## Part 1: C# Mod Changes

### 1.1 `PerimeterManager.cs` — replace single Active with sorted list

```csharp
public static class PerimeterManager
{
    private static readonly List<PerimeterData> _zones = new List<PerimeterData>();

    public static IReadOnlyList<PerimeterData> Zones => _zones;

    // Focused = highest priority zone (tile window follows this one)
    public static PerimeterData Focused => _zones.Count > 0 ? _zones[0] : null;

    public static string Place(string id, int x1, int y1, int x2, int y2, string goal, int priority)
    {
        if (_zones.Count >= 5) return "rejected_zone_cap";
        _zones.Add(new PerimeterData
        {
            Id = id, X1 = x1, Y1 = y1, X2 = x2, Y2 = y2,
            Goal = goal, Priority = priority,
        });
        _zones.Sort((a, b) => b.Priority.CompareTo(a.Priority));
        return "placed";
    }

    public static void Abandon(string id)
    {
        _zones.RemoveAll(z => z.Id == id);
        // already sorted; no re-sort needed after removal
    }

    public static object Serialize()
    {
        return _zones.Select(z => new
        {
            id       = z.Id,
            goal     = z.Goal,
            bounds   = new { x1 = z.X1, y1 = z.Y1, x2 = z.X2, y2 = z.Y2 },
            priority = z.Priority,
            status   = "active",
        }).ToList();
    }
}

public class PerimeterData
{
    public string Id;
    public int X1, Y1, X2, Y2;
    public string Goal;
    public int Priority;
}
```

### 1.2 `StateSerializer.cs` — `perimeter` → `zones` array; tile window follows Focused

Replace `perimeter = TryGet(...)` with:

```csharp
zones = TryGet("zones", GetZones, new List<object>()),
```

```csharp
private static object GetZones()
{
    return PerimeterManager.Serialize();
}
```

Tile window `GetTiles()` — change `PerimeterManager.Active` reference to `PerimeterManager.Focused`:

```csharp
var p = PerimeterManager.Focused;  // was: PerimeterManager.Active
if (p != null)
{
    const int PAD = 5;
    wx = p.X1 - PAD;
    wy = p.Y1 - PAD;
    ex = p.X2 + PAD;
    ey = p.Y2 + PAD;
}
```

### 1.3 `ActionExecutor.cs` — updated action handlers

```csharp
case "place_perimeter":
{
    string id   = cmd.GetString("id") ?? Guid.NewGuid().ToString("N").Substring(0, 8);
    int x1      = cmd.GetInt("x1"), y1 = cmd.GetInt("y1");
    int x2      = cmd.GetInt("x2"), y2 = cmd.GetInt("y2");
    string goal = cmd.GetString("goal") ?? "unknown";
    int priority = cmd.GetInt("priority");
    if (priority < 1 || priority > 9) priority = 5;
    string result = PerimeterManager.Place(id, x1, y1, x2, y2, goal, priority);
    return result;
}

case "abandon_perimeter":
{
    string id = cmd.GetString("id") ?? "";
    PerimeterManager.Abandon(id);
    return "abandoned";
}
```

### 1.4 `ActionCommand.cs` — add new fields

Add to `ActionCommand`:
```csharp
public string id       { get; set; }
public int    priority { get; set; }
```

(Existing `x1`, `y1`, `x2`, `y2`, `goal` fields remain unchanged.)

---

## Part 2: Python Agent Changes

### 2.1 `src/agent/perimeter.py` — SpatialLedger multi-zone

#### Data types

```python
@dataclass
class ActiveZone:
    id: str
    goal: str
    bounds: dict          # {x1, y1, x2, y2}
    priority: int         # 1–9
    blueprint_id: str
    cycle_started: int
    task_board: TaskBoard | None = None

@dataclass
class LedgerEntry:
    id: str
    goal: str
    blueprint_id: str
    bounds: dict
    status: str           # "complete" | "abandoned"
    cycle_started: int
    cycle_ended: int
    priority: int
```

#### SpatialLedger

```python
class SpatialLedger:
    MAX_ZONES = 5

    def __init__(self, blueprint_library: BlueprintLibrary | None = None) -> None:
        self.zones: list[ActiveZone] = []   # sorted priority desc
        self.history: list[LedgerEntry] = []
        self._library = blueprint_library or BlueprintLibrary()
        self._autocomplete_pending: list[str] = []  # zone IDs to send abandon_perimeter for

    @property
    def focused(self) -> ActiveZone | None:
        return self.zones[0] if self.zones else None

    @property
    def autocomplete_pending(self) -> list[str]:
        return list(self._autocomplete_pending)

    def clear_autocomplete(self, zone_id: str) -> None:
        self._autocomplete_pending = [z for z in self._autocomplete_pending if z != zone_id]

    def validate_place(
        self, x1: int, y1: int, x2: int, y2: int, goal: str, priority: int
    ) -> tuple[bool, str]:
        """
        Returns (True, blueprint_id) if placement is valid.
        Returns (False, reason) if rejected.
        """
        if len(self.zones) >= self.MAX_ZONES:
            return False, f"zone cap reached ({self.MAX_ZONES} active)"

        w = x2 - x1
        h = y2 - y1

        blueprint = self._library.select(goal, w, h)
        if blueprint is None:
            return False, (
                f"no blueprint matches goal='{goal}' with size {w}x{h}. "
                f"Minimum for '{goal}': check goals table in system prompt."
            )

        # 1-tile padding: no cell may touch zone boundary
        max_rel_x = w - 2
        max_rel_y = h - 2
        for entry in blueprint.get("buildings", []) + blueprint.get("dig_required", []):
            if entry["rel_x"] > max_rel_x or entry["rel_y"] > max_rel_y:
                return False, (
                    f"blueprint '{blueprint['id']}' exceeds 1-tile padding for size {w}x{h}. "
                    f"Use a larger zone."
                )

        # Overlap check
        for zone in self.zones:
            b = zone.bounds
            if not (x2 <= b["x1"] or x1 >= b["x2"] or y2 <= b["y1"] or y1 >= b["y2"]):
                return False, f"overlaps existing zone {zone.id} ({zone.goal})"

        return True, blueprint["id"]

    def on_state(self, state: dict) -> None:
        """Call each tick with parsed state dict."""
        mod_zones = state.get("zones", [])
        storage = state.get("storage", [])
        mod_ids = {z["id"] for z in mod_zones}

        # Remove zones the mod no longer knows about
        self.zones = [z for z in self.zones if z.id in mod_ids]

        # Add new zones from mod (placed this tick)
        existing_ids = {z.id for z in self.zones}
        for mz in mod_zones:
            if mz["id"] not in existing_ids:
                bounds = mz["bounds"]
                w = bounds["x2"] - bounds["x1"]
                h = bounds["y2"] - bounds["y1"]
                blueprint = self._library.select(mz["goal"], w, h)
                blueprint_id = blueprint["id"] if blueprint else ""
                self.zones.append(ActiveZone(
                    id=mz["id"],
                    goal=mz["goal"],
                    bounds=bounds,
                    priority=mz["priority"],
                    blueprint_id=blueprint_id,
                    cycle_started=state.get("cycle", 0),
                ))

        # Sort by priority desc
        self.zones.sort(key=lambda z: z.priority, reverse=True)

        # Update task boards; detect auto-complete
        for zone in self.zones:
            blueprint = self._library.get(zone.blueprint_id) if zone.blueprint_id else None
            if blueprint:
                zone.task_board = _build_task_board(blueprint, state, zone.bounds, storage)
            if zone.task_board and zone.task_board.pct >= 100.0:
                self._archive("complete", zone.id, state.get("cycle", 0))
                self._autocomplete_pending.append(zone.id)

        # Remove auto-completed zones from active list
        completed_ids = set(self._autocomplete_pending)
        self.zones = [z for z in self.zones if z.id not in completed_ids]

    def on_abandon(self, zone_id: str, cycle: int) -> None:
        zone = next((z for z in self.zones if z.id == zone_id), None)
        if zone:
            self._archive("abandoned", zone_id, cycle)
            self.zones = [z for z in self.zones if z.id != zone_id]

    def _archive(self, status: str, zone_id: str, cycle: int) -> None:
        zone = next((z for z in self.zones if z.id == zone_id), None)
        if zone:
            self.history.append(LedgerEntry(
                id=zone.id, goal=zone.goal, blueprint_id=zone.blueprint_id,
                bounds=zone.bounds, status=status,
                cycle_started=zone.cycle_started, cycle_ended=cycle,
                priority=zone.priority,
            ))

    def format_context(self) -> str:
        lines: list[str] = []
        if not self.zones:
            lines.append("No active zones — use place_perimeter to declare a build zone.")
        else:
            lines.append(f"ZONES ({len(self.zones)} active):")
            for i, zone in enumerate(self.zones):
                b = zone.bounds
                tb = zone.task_board
                pct = f"{tb.pct:.0f}%" if tb else "loading"
                steps = f"{tb.completed}/{tb.total} steps ({pct})" if tb else pct
                lines.append(
                    f"[P{zone.priority}] {zone.goal} @ "
                    f"({b['x1']},{b['y1']})-({b['x2']},{b['y2']}) — {steps}"
                )
                if i == 0 and tb:  # focused zone gets full detail
                    for t in tb.next_tasks[:3]:
                        lines.append(f"  Next: {t}")
                    for pr in tb.prerequisites[:2]:
                        lines.append(f"  Needs: {pr}")
                    if not zone.blueprint_id:
                        lines.append("  WARNING: no blueprint matched. Abandon and retry with correct size.")

        if self.history:
            recent = [f"{e.goal}(c{e.cycle_started}-{e.cycle_ended})" for e in self.history[-3:]]
            lines.append(f"Completed zones: {len(self.history)} ({', '.join(recent)})")

        return "\n".join(lines)
```

### 2.2 `src/agent/protocol.py`

Update `place_perimeter` builder and add `id` + `priority`:

```python
def build_place_perimeter(
    id: str, x1: int, y1: int, x2: int, y2: int, goal: str, priority: int
) -> dict:
    return {
        "type": "action", "action": "place_perimeter",
        "id": id, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "goal": goal, "priority": priority,
    }

def build_abandon_perimeter(zone_id: str) -> dict:
    return {"type": "action", "action": "abandon_perimeter", "id": zone_id}
```

### 2.3 `src/agent/runner.py`

- `SpatialLedger` instantiated at startup (already exists — update to use new API)
- Each tick after `ledger.on_state(state)`: iterate `ledger.autocomplete_pending`, send `abandon_perimeter(id)` for each, call `ledger.clear_autocomplete(id)`
- Before forwarding any `place_perimeter` action to mod: call `ledger.validate_place(...)`. If `(False, reason)` → do not forward; inject `! ZONE REJECTED: <reason>` into next tick's prompt state string instead.

### 2.4 `src/agent/llm.py` — system prompt update

Replace the current `## Spatial Perimeter System` section with:

```
## Spatial Perimeter System
Declare focused build zones. Up to 5 concurrent zones, each with its own task board.

place_perimeter: declare a new build zone.
  Required fields: id (generate a random 8-char hex), x1, y1, x2, y2, goal, priority (1-9, 9=highest)
  COORDINATE CONVENTION: x2/y2 are EXCLUSIVE. Interior cells: x1..x2-1, y1..y2-1.
  Example: 12-wide zone at x=115 → x1=115, x2=127. Width = x2-x1 = 12.

abandon_perimeter: cancel a specific zone by id.
  Required fields: id

Minimum zone sizes (x2-x1 >= w, y2-y1 >= h):
  "survival" / "base_camp": 12 wide x 8 tall
  "oxygen_production":       8 wide x 6 tall

CRITICAL: If prompt shows "ZONE REJECTED: ..." — read the reason and fix before retrying.
If "no blueprint matched" — increase the zone dimensions to meet minimum size.
If "overlaps existing zone" — move the zone to a non-overlapping area.

When a zone has an active task board: follow it exactly. Dig listed cells first, then
place buildings in listed order. Do NOT place_perimeter if already at 5 active zones.
```

### 2.5 `data/blueprints/base-camp-v1.json`

Bump `min_size` from `{"w": 10, "h": 6}` to `{"w": 12, "h": 8}`. No other changes needed —
all existing `rel_x` values (max 9) satisfy the 1-tile padding rule for a 12-wide zone
(max allowed = 12-2 = 10 > 9 ✓). All `rel_y` values (max 3) satisfy it for an 8-tall zone
(max allowed = 8-2 = 6 > 3 ✓).

---

## State Payload

```json
{
  "zones": [
    {
      "id": "a3f9c1e2",
      "goal": "survival",
      "bounds": {"x1": 115, "y1": 198, "x2": 127, "y2": 206},
      "priority": 9,
      "status": "active"
    },
    {
      "id": "b7d2e4f1",
      "goal": "oxygen_production",
      "bounds": {"x1": 130, "y1": 200, "x2": 138, "y2": 206},
      "priority": 6,
      "status": "active"
    }
  ]
}
```

Empty array `[]` when no zones active.

---

## Prompt Context Example

```
ZONES (2 active):
[P9] survival @ (115,198)-(127,206) — 3/9 steps (33%)
  Next: Dig (115,198)
  Next: Dig (116,198)
  Next: Place Outhouse at (115,198)
  Needs: 50kg Dirt (have 12kg)
[P6] oxygen_production @ (130,200)-(138,206) — 0/15 steps (0%)
Completed zones: 1 (base_camp(c1-4))
```

---

## Files Changed

| File | Change |
|------|--------|
| `mod/ONIBridge/src/PerimeterManager.cs` | Replace single `Active` with `List<PerimeterData>`, sort by priority, `Focused` property, `Serialize()` returns array |
| `mod/ONIBridge/src/StateSerializer.cs` | `perimeter` → `zones` array field; tile window uses `Focused` |
| `mod/ONIBridge/src/ActionExecutor.cs` | `place_perimeter` adds `id` + `priority`; `abandon_perimeter` adds `id` |
| `mod/ONIBridge/src/ActionCommand.cs` | Add `id` (string), `priority` (int) fields |
| `src/agent/perimeter.py` | `SpatialLedger` multi-zone; `validate_place()`; `format_context()` multi-zone output |
| `src/agent/protocol.py` | `build_place_perimeter` adds `id`+`priority`; `build_abandon_perimeter` adds `id` |
| `src/agent/runner.py` | Multi-zone autocomplete loop; pre-flight `validate_place()` before forwarding to mod |
| `src/agent/llm.py` | System prompt: multi-zone explanation, updated min sizes, exclusive bounds, id generation |
| `data/blueprints/base-camp-v1.json` | `min_size` → `{w:12, h:8}` |

---

## Out of Scope

- Visual overlay rendering in-game
- Zone pause/resume state
- Priority auto-adjustment based on telemetry or alerts
- Proximity-weighted priority
- Automated blueprint generation
- Inter-zone dependency graph
