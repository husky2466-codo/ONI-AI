# Spatial Perimeter System v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the spatial perimeter system from single-zone to multi-zone (max 5 concurrent), each with priority 1–9, Python-side placement validation, and exclusive coordinate convention enforced throughout.

**Architecture:** `SpatialLedger` drops `active: ActivePerimeter | None` and gains `zones: list[ActiveZone]` sorted by priority descending. Python validates placement before the action reaches the mod (blueprint match, overlap, padding, cap). The C# mod replaces its single `Active` with a `List<PerimeterData>` sorted by priority; `Focused` (highest priority) drives the tile window. The state payload field renames from `perimeter` (object|null) to `zones` (array).

**Tech Stack:** Python 3.14, pytest, C# .NET 4.7.1, Newtonsoft.Json

---

## File Map

| File | Change |
|------|--------|
| `src/agent/perimeter.py` | Replace `ActivePerimeter` + single-zone `SpatialLedger` with `ActiveZone` + multi-zone `SpatialLedger` + `validate_place()` |
| `src/agent/protocol.py` | `build_place_perimeter` gains `id`+`priority`; `build_abandon_perimeter` gains `id` |
| `src/agent/runner.py` | Multi-zone autocomplete loop; `validate_place()` pre-flight; remove old single-zone block |
| `src/agent/llm.py` | System prompt perimeter section updated: multi-zone, exclusive bounds, updated min sizes |
| `data/blueprints/base-camp-v1.json` | `min_size` → `{w:12, h:8}` |
| `mod/ONIBridge/src/PerimeterManager.cs` | Single `Active` → `List<PerimeterData>` sorted by priority; `Focused` property; `Serialize()` returns array |
| `mod/ONIBridge/src/StateSerializer.cs` | `perimeter` field → `zones`; `GetTiles()` uses `Focused` |
| `mod/ONIBridge/src/ActionExecutor.cs` | `place_perimeter` adds `id`+`priority`; `abandon_perimeter` adds `id` |
| `mod/ONIBridge/src/ActionCommand.cs` | Add `Id` (string) field; `GetString` handles `"id"` |
| `tests/agent/test_perimeter_v2.py` | New test file for `SpatialLedger` multi-zone and `validate_place()` |

---

## Task 1: Multi-zone SpatialLedger + validate_place (Python)

**Files:**
- Modify: `src/agent/perimeter.py` (full rewrite of `SpatialLedger`, add `ActiveZone` dataclass)
- Create: `tests/agent/test_perimeter_v2.py`

### Context

The current `perimeter.py` has:
- `ActivePerimeter` class (replace with `ActiveZone` dataclass)
- `SpatialLedger` with `self.active: ActivePerimeter | None`, `self._autocomplete_pending: bool`
- `on_state(state)` reads `state.get("perimeter")` (single object)
- `clear_autocomplete()` takes no args
- `on_abandon(cycle)` takes no zone id
- `format_context()` shows single zone

Keep everything else in `perimeter.py` unchanged: `LedgerEntry`, `TaskBoard`, `_topological_sort`, `_cells_needing_dig`, `_build_task_board`, `PrerequisiteResolver`, `BlueprintLibrary`.

- [ ] **Step 1: Write failing tests**

Create `tests/agent/test_perimeter_v2.py`:

```python
# tests/agent/test_perimeter_v2.py
import pytest
from src.agent.perimeter import SpatialLedger, BlueprintLibrary


def _make_library():
    """BlueprintLibrary backed by the real base-camp-v1 blueprint."""
    return BlueprintLibrary("data/blueprints")


def _make_state(zones=None, buildings=None, cycle=1):
    return {
        "cycle": cycle,
        "zones": zones or [],
        "buildings": buildings or [],
        "storage": [],
        "tiles": {},
    }


def _zone_payload(id="aaa", goal="survival", x1=115, y1=198, x2=127, y2=206, priority=9):
    return {
        "id": id,
        "goal": goal,
        "bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "priority": priority,
        "status": "active",
    }


# ---------------------------------------------------------------------------
# validate_place
# ---------------------------------------------------------------------------

def test_validate_place_valid_returns_blueprint_id():
    ledger = SpatialLedger(_make_library())
    ok, result = ledger.validate_place(115, 198, 127, 206, "survival", 9)
    assert ok is True
    assert result == "base-camp-v1"


def test_validate_place_too_small_returns_false():
    ledger = SpatialLedger(_make_library())
    # 10x6 is smaller than min_size 12x8
    ok, result = ledger.validate_place(115, 198, 125, 204, "survival", 9)
    assert ok is False
    assert "no blueprint" in result.lower()


def test_validate_place_unknown_goal_returns_false():
    ledger = SpatialLedger(_make_library())
    ok, result = ledger.validate_place(115, 198, 127, 206, "unknown_goal", 9)
    assert ok is False
    assert "no blueprint" in result.lower()


def test_validate_place_overlap_returns_false():
    ledger = SpatialLedger(_make_library())
    # Add a zone that occupies 115-127, 198-206
    ledger.on_state(_make_state(zones=[_zone_payload()]))
    # Try to place overlapping zone
    ok, result = ledger.validate_place(120, 200, 132, 208, "survival", 5)
    assert ok is False
    assert "overlap" in result.lower()


def test_validate_place_cap_reached_returns_false():
    ledger = SpatialLedger(_make_library())
    zones = [_zone_payload(id=str(i), x1=i*20, x2=i*20+12, priority=9-i) for i in range(5)]
    ledger.on_state(_make_state(zones=zones))
    ok, result = ledger.validate_place(200, 198, 212, 206, "survival", 1)
    assert ok is False
    assert "cap" in result.lower()


# ---------------------------------------------------------------------------
# on_state — multi-zone sync
# ---------------------------------------------------------------------------

def test_on_state_adds_zone_from_mod():
    ledger = SpatialLedger(_make_library())
    ledger.on_state(_make_state(zones=[_zone_payload()]))
    assert len(ledger.zones) == 1
    assert ledger.zones[0].id == "aaa"
    assert ledger.zones[0].goal == "survival"
    assert ledger.zones[0].priority == 9


def test_on_state_two_zones_sorted_by_priority():
    ledger = SpatialLedger(_make_library())
    zones = [
        _zone_payload(id="low", x1=115, x2=127, priority=3),
        _zone_payload(id="high", x1=140, x2=152, priority=8),
    ]
    ledger.on_state(_make_state(zones=zones))
    assert ledger.zones[0].id == "high"
    assert ledger.zones[1].id == "low"


def test_on_state_focused_is_highest_priority():
    ledger = SpatialLedger(_make_library())
    zones = [
        _zone_payload(id="a", x1=115, x2=127, priority=5),
        _zone_payload(id="b", x1=140, x2=152, priority=9),
    ]
    ledger.on_state(_make_state(zones=zones))
    assert ledger.focused.id == "b"


def test_on_state_empty_zones_clears_list():
    ledger = SpatialLedger(_make_library())
    ledger.on_state(_make_state(zones=[_zone_payload()]))
    ledger.on_state(_make_state(zones=[]))
    assert ledger.zones == []
    assert ledger.focused is None


def test_on_state_zone_removed_by_mod_clears_from_list():
    ledger = SpatialLedger(_make_library())
    ledger.on_state(_make_state(zones=[_zone_payload(id="aaa"), _zone_payload(id="bbb", x1=140, x2=152)]))
    assert len(ledger.zones) == 2
    # Mod removes "aaa"
    ledger.on_state(_make_state(zones=[_zone_payload(id="bbb", x1=140, x2=152)]))
    assert len(ledger.zones) == 1
    assert ledger.zones[0].id == "bbb"


# ---------------------------------------------------------------------------
# autocomplete_pending
# ---------------------------------------------------------------------------

def test_autocomplete_pending_initially_empty():
    ledger = SpatialLedger(_make_library())
    assert ledger.autocomplete_pending == []


def test_clear_autocomplete_removes_zone_id():
    ledger = SpatialLedger(_make_library())
    ledger._autocomplete_pending.append("aaa")
    ledger.clear_autocomplete("aaa")
    assert ledger.autocomplete_pending == []


# ---------------------------------------------------------------------------
# on_abandon
# ---------------------------------------------------------------------------

def test_on_abandon_archives_and_removes_zone():
    ledger = SpatialLedger(_make_library())
    ledger.on_state(_make_state(zones=[_zone_payload(id="aaa")]))
    ledger.on_abandon("aaa", cycle=3)
    assert len(ledger.zones) == 0
    assert len(ledger.history) == 1
    assert ledger.history[0].status == "abandoned"
    assert ledger.history[0].id == "aaa"


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

def test_format_context_no_zones():
    ledger = SpatialLedger(_make_library())
    ctx = ledger.format_context()
    assert "No active zones" in ctx


def test_format_context_shows_zone_count_and_priority():
    ledger = SpatialLedger(_make_library())
    ledger.on_state(_make_state(zones=[_zone_payload(id="aaa", priority=9)]))
    ctx = ledger.format_context()
    assert "ZONES (1 active)" in ctx
    assert "[P9]" in ctx
    assert "survival" in ctx


def test_format_context_two_zones_focused_first():
    ledger = SpatialLedger(_make_library())
    zones = [
        _zone_payload(id="lo", x1=115, x2=127, priority=3),
        _zone_payload(id="hi", x1=140, x2=152, priority=8),
    ]
    ledger.on_state(_make_state(zones=zones))
    ctx = ledger.format_context()
    lines = ctx.split("\n")
    # First zone line after header should be P8
    zone_lines = [l for l in lines if l.startswith("[P")]
    assert zone_lines[0].startswith("[P8]")
    assert zone_lines[1].startswith("[P3]")


def test_format_context_history_shown():
    ledger = SpatialLedger(_make_library())
    ledger.on_state(_make_state(zones=[_zone_payload(id="done")]))
    ledger.on_abandon("done", cycle=5)
    ctx = ledger.format_context()
    assert "Completed zones: 1" in ctx
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
pytest tests/agent/test_perimeter_v2.py -v 2>&1 | head -40
```

Expected: ImportError or AttributeError — `validate_place`, multi-zone API not yet on `SpatialLedger`.

- [ ] **Step 3: Replace ActivePerimeter with ActiveZone dataclass in perimeter.py**

In `src/agent/perimeter.py`, replace the `ActivePerimeter` class (lines 208–241) and the `SpatialLedger` class (lines 321–442) entirely. Keep all other code (`LedgerEntry`, `TaskBoard`, `_topological_sort`, `_cells_needing_dig`, `_build_task_board`, `PrerequisiteResolver`, `BlueprintLibrary`) unchanged.

Replace `ActivePerimeter` class with:

```python
# ---------------------------------------------------------------------------
# ActiveZone
# ---------------------------------------------------------------------------

@dataclass
class ActiveZone:
    id: str
    goal: str
    bounds: dict          # {x1, y1, x2, y2}
    priority: int         # 1–9
    blueprint_id: str
    cycle_started: int
    task_board: TaskBoard | None = None
```

Also update `LedgerEntry` to add `priority` field. Find:
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
```

Replace with:
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
    priority: int = 5
```

- [ ] **Step 4: Replace SpatialLedger with multi-zone implementation**

Replace the entire `SpatialLedger` class (from `class SpatialLedger:` through the end of the file) with:

```python
# ---------------------------------------------------------------------------
# SpatialLedger
# ---------------------------------------------------------------------------

class SpatialLedger:
    MAX_ZONES = 5

    def __init__(self, blueprint_library: BlueprintLibrary | None = None) -> None:
        self.zones: list[ActiveZone] = []        # sorted priority desc
        self.history: list[LedgerEntry] = []
        self._library = blueprint_library or BlueprintLibrary()
        self._autocomplete_pending: list[str] = []  # zone IDs runner must send abandon_perimeter for

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
        Validate a place_perimeter request before forwarding to the mod.
        Returns (True, blueprint_id) if valid.
        Returns (False, reason_string) if rejected.
        """
        if len(self.zones) >= self.MAX_ZONES:
            return False, f"zone cap reached ({self.MAX_ZONES} active)"

        w = x2 - x1
        h = y2 - y1

        blueprint = self._library.select(goal, w, h)
        if blueprint is None:
            return False, (
                f"no blueprint matches goal='{goal}' with size {w}x{h}. "
                f"Check minimum sizes in system prompt and increase zone dimensions."
            )

        # 1-tile padding: no blueprint cell may touch the zone boundary edge
        max_rel_x = w - 2
        max_rel_y = h - 2
        for entry in blueprint.get("buildings", []) + blueprint.get("dig_required", []):
            if entry["rel_x"] > max_rel_x or entry["rel_y"] > max_rel_y:
                return False, (
                    f"blueprint '{blueprint['id']}' exceeds 1-tile padding for size {w}x{h}. "
                    f"Use a larger zone."
                )

        # Overlap check against existing active zones
        for zone in self.zones:
            b = zone.bounds
            if not (x2 <= b["x1"] or x1 >= b["x2"] or y2 <= b["y1"] or y1 >= b["y2"]):
                return False, f"overlaps existing zone {zone.id} ({zone.goal})"

        return True, blueprint["id"]

    def on_state(self, state: dict) -> None:
        """Call each tick with the parsed state dict. Syncs zones from mod, updates task boards."""
        mod_zones = state.get("zones", [])
        storage = state.get("storage", [])
        mod_ids = {z["id"] for z in mod_zones}

        # Remove zones the mod no longer reports
        self.zones = [z for z in self.zones if z.id in mod_ids]

        # Add zones new in this tick
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
                    priority=mz.get("priority", 5),
                    blueprint_id=blueprint_id,
                    cycle_started=state.get("cycle", 0),
                ))
                logger.info(
                    "Zone activated: %s goal=%s blueprint=%s priority=%d",
                    mz["id"], mz["goal"], blueprint_id, mz.get("priority", 5),
                )

        # Sort by priority descending
        self.zones.sort(key=lambda z: z.priority, reverse=True)

        # Update task boards; detect auto-complete
        completed_ids: list[str] = []
        for zone in self.zones:
            blueprint = self._library.get(zone.blueprint_id) if zone.blueprint_id else None
            if blueprint:
                zone.task_board = _build_task_board(blueprint, state, zone.bounds, storage)
            if zone.task_board and zone.task_board.pct >= 100.0:
                logger.info("Zone %s auto-complete (100%%) — signalling runner", zone.id)
                self._archive("complete", zone.id, state.get("cycle", 0))
                self._autocomplete_pending.append(zone.id)
                completed_ids.append(zone.id)

        # Remove auto-completed zones
        if completed_ids:
            completed_set = set(completed_ids)
            self.zones = [z for z in self.zones if z.id not in completed_set]

    def on_abandon(self, zone_id: str, cycle: int) -> None:
        """Call when the runner explicitly sends abandon_perimeter for a zone."""
        self._archive("abandoned", zone_id, cycle)
        self.zones = [z for z in self.zones if z.id != zone_id]

    def _archive(self, status: str, zone_id: str, cycle: int) -> None:
        zone = next((z for z in self.zones if z.id == zone_id), None)
        if zone:
            self.history.append(LedgerEntry(
                id=zone.id,
                goal=zone.goal,
                blueprint_id=zone.blueprint_id,
                bounds=zone.bounds,
                status=status,
                cycle_started=zone.cycle_started,
                cycle_ended=cycle,
                priority=zone.priority,
            ))

    def format_context(self) -> str:
        """Compact string for injection into the AI prompt."""
        lines: list[str] = []

        if not self.zones:
            lines.append("No active zones — use place_perimeter to declare a build zone.")
        else:
            lines.append(f"ZONES ({len(self.zones)} active):")
            for i, zone in enumerate(self.zones):
                b = zone.bounds
                tb = zone.task_board
                if tb:
                    pct_str = f"{tb.pct:.0f}%"
                    steps_str = f"{tb.completed}/{tb.total} steps ({pct_str})"
                else:
                    steps_str = "loading"
                lines.append(
                    f"[P{zone.priority}] {zone.goal} @ "
                    f"({b['x1']},{b['y1']})-({b['x2']},{b['y2']}) — {steps_str}"
                )
                if i == 0 and tb:  # focused zone gets full detail
                    for t in tb.next_tasks[:3]:
                        lines.append(f"  Next: {t}")
                    for pr in tb.prerequisites[:2]:
                        lines.append(f"  Needs: {pr}")
                if i == 0 and not zone.blueprint_id:
                    lines.append("  WARNING: no blueprint matched. Abandon and retry with correct size.")

        if self.history:
            recent = [
                f"{e.goal}(c{e.cycle_started}-{e.cycle_ended})"
                for e in self.history[-3:]
            ]
            lines.append(f"Completed zones: {len(self.history)} ({', '.join(recent)})")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot for episode logging."""
        return {
            "zones": [
                {
                    "id": z.id,
                    "goal": z.goal,
                    "bounds": z.bounds,
                    "blueprint_id": z.blueprint_id,
                    "priority": z.priority,
                    "cycle_started": z.cycle_started,
                    "task_board": {
                        "total": z.task_board.total,
                        "completed": z.task_board.completed,
                        "pct": z.task_board.pct,
                    } if z.task_board else None,
                }
                for z in self.zones
            ],
            "history": [
                {
                    "id": e.id,
                    "goal": e.goal,
                    "status": e.status,
                    "cycle_started": e.cycle_started,
                    "cycle_ended": e.cycle_ended,
                    "priority": e.priority,
                }
                for e in self.history
            ],
        }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/agent/test_perimeter_v2.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Run full agent test suite**

```bash
pytest tests/agent/ -v
```

Expected: all existing tests still pass (no regressions).

- [ ] **Step 7: Commit**

```bash
git add src/agent/perimeter.py tests/agent/test_perimeter_v2.py
git commit -m "feat: multi-zone SpatialLedger with validate_place, priority sorting, auto-complete per zone"
```

---

## Task 2: Update protocol.py — place_perimeter and abandon_perimeter signatures

**Files:**
- Modify: `src/agent/protocol.py:81-86`
- Test: `tests/agent/test_protocol.py`

### Context

Current `protocol.py` has:
```python
def build_place_perimeter(x1: int, y1: int, x2: int, y2: int, goal: str) -> dict[str, Any]:
    return build_action("place_perimeter", x1=x1, y1=y1, x2=x2, y2=y2, goal=goal)

def build_abandon_perimeter() -> dict[str, Any]:
    return build_action("abandon_perimeter")
```

Both need `id`. `build_place_perimeter` also needs `priority`.

- [ ] **Step 1: Write failing tests**

Add to `tests/agent/test_protocol.py` (read the file first to find the end, then append):

```python
def test_build_place_perimeter_includes_id_and_priority():
    from src.agent.protocol import build_place_perimeter
    action = build_place_perimeter(
        id="abc12345", x1=115, y1=198, x2=127, y2=206, goal="survival", priority=9
    )
    assert action["action"] == "place_perimeter"
    assert action["id"] == "abc12345"
    assert action["priority"] == 9
    assert action["x1"] == 115
    assert action["goal"] == "survival"


def test_build_abandon_perimeter_includes_id():
    from src.agent.protocol import build_abandon_perimeter
    action = build_abandon_perimeter(zone_id="abc12345")
    assert action["action"] == "abandon_perimeter"
    assert action["id"] == "abc12345"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agent/test_protocol.py::test_build_place_perimeter_includes_id_and_priority tests/agent/test_protocol.py::test_build_abandon_perimeter_includes_id -v
```

Expected: FAIL — `build_place_perimeter` doesn't accept `id` or `priority`; `build_abandon_perimeter` takes no args.

- [ ] **Step 3: Update build_place_perimeter and build_abandon_perimeter**

In `src/agent/protocol.py`, replace:
```python
def build_place_perimeter(x1: int, y1: int, x2: int, y2: int, goal: str) -> dict[str, Any]:
    return build_action("place_perimeter", x1=x1, y1=y1, x2=x2, y2=y2, goal=goal)


def build_abandon_perimeter() -> dict[str, Any]:
    return build_action("abandon_perimeter")
```

With:
```python
def build_place_perimeter(
    id: str, x1: int, y1: int, x2: int, y2: int, goal: str, priority: int
) -> dict[str, Any]:
    return build_action("place_perimeter", id=id, x1=x1, y1=y1, x2=x2, y2=y2,
                        goal=goal, priority=priority)


def build_abandon_perimeter(zone_id: str) -> dict[str, Any]:
    return build_action("abandon_perimeter", id=zone_id)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agent/test_protocol.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/agent/protocol.py tests/agent/test_protocol.py
git commit -m "feat: update place_perimeter/abandon_perimeter protocol to include id and priority"
```

---

## Task 3: Update runner.py — multi-zone autocomplete and validate_place pre-flight

**Files:**
- Modify: `src/agent/runner.py` (ledger integration section, ~lines 358–376 and ~503–510)

### Context

Current runner.py has these ledger-related blocks:

**Block A** (lines ~358–376): single-zone autocomplete + auto-abandon for no-blueprint:
```python
ledger.on_state(state.data)

if ledger.autocomplete_pending:
    ledger.clear_autocomplete()
    ...
    await client.send_action(build_abandon_perimeter())

if ledger.active is not None and not ledger.active.blueprint_id:
    if not getattr(ledger.active, '_abandon_sent', False):
        ledger.active._abandon_sent = True
        await client.send_action(build_abandon_perimeter())
    continue
```

**Block B** (lines ~503–510): single-zone place_perimeter block:
```python
if candidate.get("action") == "place_perimeter" and ledger.active is not None:
    logger.info("  -> runner blocked place_perimeter (perimeter already active)")
    _validation_result = "blocked"
    _validation_reason = "perimeter already active"
    candidate = build_no_op()
```

Also: `build_abandon_perimeter` import (line 40) and `ledger.active` references (lines 370, 506) need updating.

Also: the log entry builder at line ~568 uses `action.get('goal','')` — that still works, no change needed.

- [ ] **Step 1: Update import line**

In `src/agent/runner.py`, find:
```python
from src.agent.protocol import build_abandon_perimeter, build_no_op
```

Replace with:
```python
from src.agent.protocol import build_abandon_perimeter, build_no_op, build_place_perimeter
```

- [ ] **Step 2: Replace Block A — multi-zone autocomplete**

Find and replace the entire block from `# Update spatial ledger each tick` through the `continue` after the auto-abandon block:

```python
            # Update spatial ledger each tick
            ledger.on_state(state.data)

            # Auto-complete: send abandon_perimeter for each zone that hit 100%
            if ledger.autocomplete_pending:
                ledger.clear_autocomplete()
                logger.info("Perimeter auto-complete — sending abandon_perimeter")
                await client.send_action(build_abandon_perimeter())
                relay.pending_action = build_abandon_perimeter()

            # Auto-abandon: perimeter placed but no blueprint matched — useless, clear it.
            # Mark the blueprint_id as sentinel so we don't spam abandon every tick
            # while waiting for the mod to confirm the abandon and clear its state.
            if ledger.active is not None and not ledger.active.blueprint_id:
                if not getattr(ledger.active, '_abandon_sent', False):
                    logger.warning("Perimeter has no matching blueprint — auto-abandoning")
                    ledger.active._abandon_sent = True
                    await client.send_action(build_abandon_perimeter())
                    relay.pending_action = build_abandon_perimeter()
                continue
```

With:

```python
            # Update spatial ledger each tick
            ledger.on_state(state.data)

            # Auto-complete: for each zone that hit 100%, send abandon_perimeter(id)
            for _zone_id in ledger.autocomplete_pending:
                logger.info("Zone %s auto-complete (100%%) — sending abandon_perimeter", _zone_id)
                _abandon = build_abandon_perimeter(_zone_id)
                await client.send_action(_abandon)
                relay.pending_action = _abandon
                ledger.clear_autocomplete(_zone_id)
```

- [ ] **Step 3: Replace Block B — validate_place pre-flight**

Find and replace:
```python
                # Hard block: never send place_perimeter when one is already active.
                # The mod rejects it anyway, but this prevents the agent from wasting
                # every tick retrying it instead of doing actual work.
                if candidate.get("action") == "place_perimeter" and ledger.active is not None:
                    logger.info("  -> runner blocked place_perimeter (perimeter already active)")
                    _validation_result = "blocked"
                    _validation_reason = "perimeter already active"
                    candidate = build_no_op()
```

With:
```python
                # Validate place_perimeter before forwarding to mod.
                # Checks: blueprint match, overlap, padding, zone cap.
                if candidate.get("action") == "place_perimeter":
                    import uuid
                    _zid = candidate.get("id") or uuid.uuid4().hex[:8]
                    candidate["id"] = _zid
                    _priority = int(candidate.get("priority", 5))
                    _ok, _reason = ledger.validate_place(
                        int(candidate.get("x1", 0)), int(candidate.get("y1", 0)),
                        int(candidate.get("x2", 0)), int(candidate.get("y2", 0)),
                        str(candidate.get("goal", "")), _priority,
                    )
                    if not _ok:
                        logger.info("  -> runner blocked place_perimeter: %s", _reason)
                        _validation_result = "blocked"
                        _validation_reason = _reason
                        candidate = build_no_op()
                        # Inject rejection reason into next tick via ledger warning field
                        ledger._last_rejection = _reason
                    else:
                        ledger._last_rejection = None
```

- [ ] **Step 4: Inject rejection reason into prompt context**

In `SpatialLedger.format_context()` in `src/agent/perimeter.py`, add after the `if not self.zones:` block (at the very top of `format_context`), before the `lines` list is built:

Find the start of `format_context`:
```python
    def format_context(self) -> str:
        """Compact string for injection into the AI prompt."""
        lines: list[str] = []
```

Replace with:
```python
    def format_context(self) -> str:
        """Compact string for injection into the AI prompt."""
        lines: list[str] = []
        # Show rejection reason from last blocked place_perimeter (set by runner)
        rejection = getattr(self, "_last_rejection", None)
        if rejection:
            lines.append(f"! ZONE REJECTED: {rejection}")
```

- [ ] **Step 5: Run full agent test suite**

```bash
pytest tests/agent/ -v
```

Expected: all pass. No tests cover the runner internals for perimeter directly, but existing runner validation tests must still pass.

- [ ] **Step 6: Commit**

```bash
git add src/agent/runner.py src/agent/perimeter.py
git commit -m "feat: runner multi-zone autocomplete loop and validate_place pre-flight"
```

---

## Task 4: Update system prompt in llm.py

**Files:**
- Modify: `src/agent/llm.py` (Spatial Perimeter System section, ~lines 163–176)
- Test: `tests/agent/test_llm_wiki.py`

### Context

Current system prompt section:
```
## Spatial Perimeter System
Use perimeters to declare a focused construction zone. One active perimeter at a time.
  place_perimeter: declare a build zone with a goal. The task board will tell you what to build.
  abandon_perimeter: cancel current perimeter (auto-abandon happens at 100% completion).

COORDINATE CONVENTION: x2 and y2 are EXCLUSIVE — cells x1 through x2-1 and y1 through y2-1 are inside.
  Example: 10-wide zone starting at x=115 → x1=115, x2=125 (NOT 124). x2-x1 must equal the width.

Goals and MINIMUM sizes:
  "survival" / "base_camp": min 10 wide x 6 tall → x2-x1=10, y2-y1=6 (e.g. x1=115,x2=125,y1=198,y2=204)
  "oxygen_production":       min 6 wide x 4 tall  → x2-x1=6,  y2-y1=4

CRITICAL: If the Spatial Ledger says "no blueprint matched" — abandon the perimeter immediately
and place a new one at the correct minimum size.

When a perimeter is active with a loaded task board, follow it exactly — dig listed cells first,
then place buildings in order. Do NOT place_perimeter if one is already active.
```

- [ ] **Step 1: Write failing test**

Add to `tests/agent/test_llm_wiki.py`:

```python
def test_system_prompt_multi_zone_perimeter_section():
    """SYSTEM_PROMPT must document multi-zone, id, priority, exclusive bounds, and correct min sizes."""
    from src.agent.llm import SYSTEM_PROMPT
    assert "Up to 5" in SYSTEM_PROMPT or "5 concurrent" in SYSTEM_PROMPT
    assert "priority" in SYSTEM_PROMPT
    assert "id" in SYSTEM_PROMPT
    assert "EXCLUSIVE" in SYSTEM_PROMPT
    # Updated min sizes
    assert "12 wide" in SYSTEM_PROMPT or "x2-x1=12" in SYSTEM_PROMPT
    assert "8 tall" in SYSTEM_PROMPT or "y2-y1=8" in SYSTEM_PROMPT
    # No longer "one active perimeter at a time"
    assert "One active perimeter at a time" not in SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agent/test_llm_wiki.py::test_system_prompt_multi_zone_perimeter_section -v
```

Expected: FAIL — current prompt says "One active perimeter at a time" and has old min sizes.

- [ ] **Step 3: Replace the Spatial Perimeter System section**

In `src/agent/llm.py`, find the block starting with `## Spatial Perimeter System` through `do NOT place_perimeter if one is already active.` and replace entirely with:

```
## Spatial Perimeter System
Declare focused build zones. Up to 5 concurrent zones, each with its own task board.

place_perimeter: declare a new build zone.
  Required fields: id (generate a random 8-char hex string), x1, y1, x2, y2, goal, priority (1-9, 9=highest)
  COORDINATE CONVENTION: x2/y2 are EXCLUSIVE. Interior cells: x1..x2-1, y1..y2-1. Width = x2-x1.
  Example: 12-wide zone at x=115 → x1=115, x2=127 (NOT 126). x2-x1 must equal the width.

abandon_perimeter: cancel a specific zone.
  Required fields: id (the zone id to cancel)

Minimum zone sizes (x2-x1 >= w, y2-y1 >= h):
  "survival" / "base_camp": 12 wide x 8 tall → x2-x1=12, y2-y1=8
  "oxygen_production":       8 wide x 6 tall  → x2-x1=8,  y2-y1=6

CRITICAL: If prompt shows "! ZONE REJECTED: ..." — read the reason and fix before retrying.
  "no blueprint matched" → increase zone dimensions to meet minimum size above.
  "overlaps existing zone" → move zone to a non-overlapping area.
  "zone cap reached" → abandon a completed or low-priority zone first.

When a zone has an active task board: follow it exactly — dig listed cells first, then place
buildings in listed order. Higher priority zones are shown first in the Spatial Ledger.
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/agent/test_llm_wiki.py::test_system_prompt_multi_zone_perimeter_section -v
```

Expected: PASS.

- [ ] **Step 5: Run full agent test suite**

```bash
pytest tests/agent/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/agent/llm.py tests/agent/test_llm_wiki.py
git commit -m "feat: update system prompt for multi-zone perimeter, exclusive bounds, new min sizes"
```

---

## Task 5: Fix base-camp-v1.json min_size

**Files:**
- Modify: `data/blueprints/base-camp-v1.json`
- Test: `tests/agent/test_perimeter_v2.py` (test_validate_place_valid_returns_blueprint_id already covers this)

### Context

Current `base-camp-v1.json`:
```json
"min_size": { "w": 10, "h": 6 }
```

Blueprint uses `rel_x` 0–9, `rel_y` 0–3. For 1-tile padding in a 12-wide zone: max allowed `rel_x = 12-2 = 10 > 9` ✓. For 8-tall zone: max allowed `rel_y = 8-2 = 6 > 3` ✓. So bumping `min_size` to 12×8 satisfies the padding rule without changing any coordinates.

- [ ] **Step 1: Bump min_size**

In `data/blueprints/base-camp-v1.json`, replace:
```json
"min_size": { "w": 10, "h": 6 },
```

With:
```json
"min_size": { "w": 12, "h": 8 },
```

- [ ] **Step 2: Verify test_validate_place_valid_returns_blueprint_id passes**

```bash
pytest tests/agent/test_perimeter_v2.py::test_validate_place_valid_returns_blueprint_id -v
```

Expected: PASS (12×8 zone passes validation and selects `base-camp-v1`).

- [ ] **Step 3: Verify too-small test also passes**

```bash
pytest tests/agent/test_perimeter_v2.py::test_validate_place_too_small_returns_false -v
```

Expected: PASS (10×6 zone no longer matches).

- [ ] **Step 4: Commit**

```bash
git add data/blueprints/base-camp-v1.json
git commit -m "fix: bump base-camp-v1 min_size to 12x8 to satisfy 1-tile padding rule"
```

---

## Task 6: C# mod — multi-zone PerimeterManager, StateSerializer, ActionExecutor, ActionCommand

**Files:**
- Modify: `mod/ONIBridge/src/PerimeterManager.cs`
- Modify: `mod/ONIBridge/src/StateSerializer.cs`
- Modify: `mod/ONIBridge/src/ActionExecutor.cs`
- Modify: `mod/ONIBridge/src/ActionCommand.cs`

**Note:** C# changes cannot be unit-tested without the game runtime. Verify by building with `dotnet build` and watching the game log after deploy.

### Context

`PerimeterManager.cs` currently has a single `Active` property and `Place(x1,y1,x2,y2,goal)` that rejects if one is already active.

`StateSerializer.cs` line 45: `perimeter = TryGet("perimeter", GetPerimeter, (object)null)`

`ActionExecutor.cs` lines 34–50: `place_perimeter` reads x1/y1/x2/y2/goal; `abandon_perimeter` calls `PerimeterManager.Abandon()` with no args.

`ActionCommand.cs` has no `Id` field; `GetString` has no `"id"` case.

- [ ] **Step 1: Replace PerimeterManager.cs**

Replace the entire contents of `mod/ONIBridge/src/PerimeterManager.cs` with:

```csharp
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// Stores all active build zones (max 5), sorted by priority descending.
    /// No game sim interaction — dupes cannot see or interact with zones.
    /// Completion detection is handled Python-side; Python sends abandon_perimeter(id)
    /// when auto-complete triggers or the agent explicitly abandons.
    /// </summary>
    public static class PerimeterManager
    {
        private static readonly List<PerimeterData> _zones = new List<PerimeterData>();

        public static IReadOnlyList<PerimeterData> Zones => _zones;

        /// <summary>Highest-priority zone — tile window follows this one.</summary>
        public static PerimeterData Focused => _zones.Count > 0 ? _zones[0] : null;

        /// <summary>
        /// Place a new zone. Returns "placed" on success, or a rejection reason string.
        /// </summary>
        public static string Place(string id, int x1, int y1, int x2, int y2, string goal, int priority)
        {
            if (_zones.Count >= 5)
            {
                Debug.LogWarning("[ONIBridge] place_perimeter rejected — zone cap (5) reached");
                return "rejected_zone_cap";
            }
            _zones.Add(new PerimeterData
            {
                Id = id,
                X1 = x1, Y1 = y1, X2 = x2, Y2 = y2,
                Goal = goal,
                Priority = priority,
            });
            _zones.Sort((a, b) => b.Priority.CompareTo(a.Priority));
            Debug.Log($"[ONIBridge] Zone placed: {id} goal={goal} priority={priority} bounds=({x1},{y1})-({x2},{y2})");
            return "placed";
        }

        /// <summary>Abandon a specific zone by id.</summary>
        public static void Abandon(string id)
        {
            int removed = _zones.RemoveAll(z => z.Id == id);
            if (removed > 0)
                Debug.Log($"[ONIBridge] Zone abandoned: {id}");
            else
                Debug.LogWarning($"[ONIBridge] abandon_perimeter: zone {id} not found");
        }

        /// <summary>Produces the zones array for the state message.</summary>
        public static object Serialize()
        {
            return _zones.Select(z => (object)new
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
}
```

- [ ] **Step 2: Update StateSerializer.cs — perimeter → zones**

In `mod/ONIBridge/src/StateSerializer.cs`, find:
```csharp
                perimeter      = TryGet("perimeter",       GetPerimeter,      (object)null),
```

Replace with:
```csharp
                zones          = TryGet("zones",           GetZones,          new System.Collections.Generic.List<object>()),
```

Find:
```csharp
        private static object GetPerimeter()
        {
            return PerimeterManager.Serialize();
        }
```

Replace with:
```csharp
        private static object GetZones()
        {
            return PerimeterManager.Serialize();
        }
```

Find in `GetTiles()`:
```csharp
            var perim = PerimeterManager.Active;
```

Replace with:
```csharp
            var perim = PerimeterManager.Focused;
```

- [ ] **Step 3: Update ActionCommand.cs — add Id field and GetString case**

In `mod/ONIBridge/src/ActionCommand.cs`, find:
```csharp
        // Perimeter goal (e.g. "oxygen_production")
        [JsonProperty("goal")]
        public string? Goal { get; set; }
```

Replace with:
```csharp
        // Perimeter zone id (Python-generated 8-char hex)
        [JsonProperty("id")]
        public string? Id { get; set; }

        // Perimeter goal (e.g. "oxygen_production")
        [JsonProperty("goal")]
        public string? Goal { get; set; }
```

In the `GetString` method, find:
```csharp
                case "goal": return Goal;
                case "tech_id": return TechId;
                case "building_id": return BuildingId;
                default: return null;
```

Replace with:
```csharp
                case "id": return Id;
                case "goal": return Goal;
                case "tech_id": return TechId;
                case "building_id": return BuildingId;
                default: return null;
```

- [ ] **Step 4: Update ActionExecutor.cs — place_perimeter and abandon_perimeter**

In `mod/ONIBridge/src/ActionExecutor.cs`, find:
```csharp
                    case "place_perimeter":
                    {
                        int x1 = cmd.GetInt("x1"), y1 = cmd.GetInt("y1");
                        int x2 = cmd.GetInt("x2"), y2 = cmd.GetInt("y2");
                        string goal = cmd.GetString("goal") ?? "unknown";
                        bool ok = PerimeterManager.Place(x1, y1, x2, y2, goal);
                        if (ok)
                            BridgeServer.Instance.SendAck(cmd.Action, true);
                        else
                            BridgeServer.Instance.SendAck(cmd.Action, false, "perimeter_already_active");
                        break;
                    }
                    case "abandon_perimeter":
                    {
                        PerimeterManager.Abandon();
                        BridgeServer.Instance.SendAck(cmd.Action, true);
                        break;
                    }
```

Replace with:
```csharp
                    case "place_perimeter":
                    {
                        string id = cmd.GetString("id") ?? System.Guid.NewGuid().ToString("N").Substring(0, 8);
                        int x1 = cmd.GetInt("x1"), y1 = cmd.GetInt("y1");
                        int x2 = cmd.GetInt("x2"), y2 = cmd.GetInt("y2");
                        string goal = cmd.GetString("goal") ?? "unknown";
                        int priority = cmd.GetInt("priority");
                        if (priority < 1 || priority > 9) priority = 5;
                        string result = PerimeterManager.Place(id, x1, y1, x2, y2, goal, priority);
                        if (result == "placed")
                            BridgeServer.Instance.SendAck(cmd.Action, true);
                        else
                            BridgeServer.Instance.SendAck(cmd.Action, false, result);
                        break;
                    }
                    case "abandon_perimeter":
                    {
                        string id = cmd.GetString("id") ?? "";
                        if (!string.IsNullOrEmpty(id))
                            PerimeterManager.Abandon(id);
                        BridgeServer.Instance.SendAck(cmd.Action, true);
                        break;
                    }
```

- [ ] **Step 5: Build the mod**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge
dotnet build
```

Expected: Build succeeded, 0 Error(s). Fix any compilation errors before proceeding.

- [ ] **Step 6: Commit**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
git add mod/ONIBridge/src/PerimeterManager.cs \
        mod/ONIBridge/src/StateSerializer.cs \
        mod/ONIBridge/src/ActionExecutor.cs \
        mod/ONIBridge/src/ActionCommand.cs
git commit -m "feat: C# mod multi-zone PerimeterManager, zones[] state field, id-targeted abandon"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|------------------|------|
| Max 5 concurrent zones, sorted priority desc | Task 1 (SpatialLedger.zones, MAX_ZONES=5) |
| Priority 1–9, 9=highest | Task 1, Task 6 |
| Lifecycle: active → complete \| abandoned only | Task 1 (on_state, on_abandon) |
| validate_place: blueprint match, overlap, padding, cap | Task 1 |
| Python-generated zone ID | Task 3 (uuid generation in runner) |
| Focused = highest-priority zone | Task 1 (focused property) |
| Tile window follows Focused | Task 6 (StateSerializer Focused) |
| format_context: focused zone full detail, others one-line | Task 1 |
| Rejection reason injected into prompt | Task 3 + Task 1 (_last_rejection) |
| place_perimeter gains id + priority | Task 2, Task 3, Task 6 |
| abandon_perimeter gains id | Task 2, Task 3, Task 6 |
| state payload: perimeter → zones array | Task 6 |
| base-camp-v1 min_size bumped to 12×8 | Task 5 |
| System prompt updated for multi-zone | Task 4 |
| Exclusive bounds documented with example | Task 4 |
| Multi-zone autocomplete loop in runner | Task 3 |

**Placeholder scan:** No TBDs. All code blocks complete.

**Type consistency:**
- `validate_place` returns `tuple[bool, str]` — used as `ok, result` in tests and `_ok, _reason` in runner ✓
- `clear_autocomplete(zone_id: str)` — called with `_zone_id` in runner ✓
- `on_abandon(zone_id: str, cycle: int)` — consistent across Task 1 test and Task 3 ✓
- `build_abandon_perimeter(zone_id: str)` in protocol → sends `{"id": zone_id}` → C# reads `cmd.GetString("id")` ✓
- `LedgerEntry.priority` added — `to_dict()` serializes it ✓
