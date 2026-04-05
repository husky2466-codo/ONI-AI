# src/agent/perimeter.py
"""
Spatial Perimeter System — Python agent side.

Components:
  SpatialLedger      — tracks active + archived perimeters across ticks
  ActivePerimeter    — wraps a live perimeter with its TaskBoard
  TaskBoard          — deterministic diff of blueprint vs. actual buildings
  PrerequisiteResolver — diffs build requirements vs. storage inventory
  BlueprintLibrary   — loads and selects blueprint templates by goal
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

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


@dataclass
class TaskBoard:
    total: int
    completed: int
    pct: float
    next_tasks: list[str]        # ordered by dependency graph
    prerequisites: list[str]     # resource deficits blocking next tasks


# ---------------------------------------------------------------------------
# TopSort helper
# ---------------------------------------------------------------------------

def _topological_sort(steps: list[dict], dependencies: dict[str, list[str]]) -> list[dict]:
    """
    Order construction steps by their dependency graph.
    Steps that have no unmet dependencies come first.
    Cycle resolution: if a cycle is detected, break it by preferring the node
    with fewer total dependents. Log a warning — cycles in blueprint JSON are data errors.
    """
    type_to_step = {s["type"]: s for s in steps}
    in_degree: dict[str, int] = {s["type"]: 0 for s in steps}
    dependents: dict[str, list[str]] = {s["type"]: [] for s in steps}

    for node, deps in dependencies.items():
        if node not in in_degree:
            continue
        for dep in deps:
            if dep in in_degree:
                in_degree[node] += 1
                dependents[dep].append(node)

    queue = sorted(
        [t for t, d in in_degree.items() if d == 0],
        key=lambda t: len(dependents.get(t, []))
    )
    result: list[dict] = []
    visited: set[str] = set()

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if current in type_to_step:
            result.append(type_to_step[current])
        for dependent in dependents.get(current, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # Any remaining steps had unresolvable dependencies (missing from remaining list
    # because they're already built, or a true cycle). Append them in original order.
    remaining_types = {s["type"] for s in steps} - visited
    if remaining_types:
        logger.warning(
            "Blueprint dependency cycle or unresolvable deps detected for: %s",
            remaining_types,
        )
    for s in steps:
        if s["type"] not in visited:
            result.append(s)

    return result


# ---------------------------------------------------------------------------
# TaskBoard builder
# ---------------------------------------------------------------------------

def _cells_needing_dig(blueprint: dict, state: dict, bounds: dict) -> list[tuple[int, int]]:
    """
    Return absolute (x, y) coords from dig_required that still have solid mass.
    Uses the tile window in state to check — cells not in the window are assumed
    uncleared (conservatively require digging).
    """
    dig_required = blueprint.get("dig_required", [])
    if not dig_required:
        return []

    origin_x = bounds["x1"]
    origin_y = bounds["y1"]

    # Build a lookup from (cx, cy) -> mass from the tile window
    tiles = state.get("tiles", {})
    tile_data = tiles.get("data", [])
    tw_x = tiles.get("x", 0)
    tw_y = tiles.get("y", 0)
    tw_w = tiles.get("w", 0)
    tile_mass: dict[tuple[int, int], float] = {}
    for idx, cell in enumerate(tile_data):
        col = idx % tw_w if tw_w else 0
        row = idx // tw_w if tw_w else 0
        cx = tw_x + col
        cy = tw_y + row
        mass = cell[1] if isinstance(cell, (list, tuple)) and len(cell) >= 2 else 0.0
        tile_mass[(cx, cy)] = float(mass)

    needs_dig = []
    not_in_window = []
    for entry in dig_required:
        ax = origin_x + entry["rel_x"]
        ay = origin_y + entry["rel_y"]
        mass = tile_mass.get((ax, ay), -1.0)
        if mass > 0:
            needs_dig.append((ax, ay))
        elif mass < 0:
            not_in_window.append((ax, ay))
        # mass == 0: confirmed open, skip

    if not_in_window:
        logger.warning(
            "_cells_needing_dig: %d dig_required cells not in tile window — "
            "tile window may not cover perimeter. First few: %s",
            len(not_in_window), not_in_window[:5]
        )

    return needs_dig


def _build_task_board(
    blueprint: dict,
    state: dict,
    bounds: dict,
    storage: list[dict],
) -> TaskBoard:
    buildings_placed = {
        (b["type"], b["x"], b["y"]) for b in state.get("buildings", [])
    }
    steps = blueprint.get("buildings", [])
    origin_x = bounds["x1"]
    origin_y = bounds["y1"]

    completed = 0
    remaining: list[dict] = []
    for step in steps:
        abs_x = origin_x + step["rel_x"]
        abs_y = origin_y + step["rel_y"]
        if (step["type"], abs_x, abs_y) in buildings_placed:
            completed += 1
        else:
            remaining.append({**step, "abs_x": abs_x, "abs_y": abs_y})

    ordered = _topological_sort(remaining, blueprint.get("dependencies", {}))

    # Dig tasks come before build tasks — show up to 3 pending digs first
    dig_cells = _cells_needing_dig(blueprint, state, bounds)
    dig_tasks = [f"Dig ({cx},{cy})" for cx, cy in dig_cells[:3]]
    build_tasks = [
        f"Place {s['type']} at ({s['abs_x']},{s['abs_y']})"
        for s in ordered[: max(0, 5 - len(dig_tasks))]
    ]
    next_tasks = dig_tasks + build_tasks

    pct = 100.0 * completed / len(steps) if steps else 0.0
    prerequisites = PrerequisiteResolver().resolve(ordered[:3], storage)

    return TaskBoard(
        total=len(steps),
        completed=completed,
        pct=pct,
        next_tasks=next_tasks,
        prerequisites=prerequisites,
    )


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


# ---------------------------------------------------------------------------
# PrerequisiteResolver
# ---------------------------------------------------------------------------

class PrerequisiteResolver:
    def resolve(self, next_steps: list[dict], storage: list[dict]) -> list[str]:
        """Returns resource deficit strings for the first 3 next steps."""
        available = self._aggregate_storage(storage)
        deficits: list[str] = []
        seen: set[str] = set()
        for step in next_steps:
            for req in step.get("requires", []):
                element = req["element"]
                need = float(req["mass_kg"])
                have = available.get(element, 0.0)
                if have < need:
                    key = f"{element}:{need}"
                    if key not in seen:
                        seen.add(key)
                        deficits.append(
                            f"Need {need:.0f}kg {element} (have {have:.0f}kg)"
                        )
        return deficits

    def _aggregate_storage(self, storage: list[dict]) -> dict[str, float]:
        totals: dict[str, float] = {}
        for container in storage:
            for item in container.get("contents", []):
                e = item["element"]
                totals[e] = totals.get(e, 0.0) + float(item["mass_kg"])
        return totals


# ---------------------------------------------------------------------------
# BlueprintLibrary
# ---------------------------------------------------------------------------

class BlueprintLibrary:
    def __init__(self, blueprint_dir: str = "data/blueprints") -> None:
        self._blueprints: dict[str, dict] = {}
        self._load(Path(blueprint_dir))

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Blueprint directory %s not found — no blueprints loaded", path)
            return
        for fp in path.glob("*.json"):
            try:
                bp = json.loads(fp.read_text())
                self._blueprints[bp["id"]] = bp
                logger.debug("Loaded blueprint: %s", bp["id"])
            except Exception as e:
                logger.warning("Failed to load blueprint %s: %s", fp, e)
        logger.info("BlueprintLibrary: loaded %d blueprints", len(self._blueprints))

    def get(self, blueprint_id: str) -> dict | None:
        return self._blueprints.get(blueprint_id)

    def select(self, goal: str, available_w: int, available_h: int) -> dict | None:
        """Select the smallest blueprint that matches the goal and fits the space."""
        candidates = [
            bp for bp in self._blueprints.values()
            if goal in bp.get("goals", [])
            and bp["min_size"]["w"] <= available_w
            and bp["min_size"]["h"] <= available_h
        ]
        return min(
            candidates,
            key=lambda b: b["min_size"]["w"] * b["min_size"]["h"],
            default=None,
        )


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
        self._last_rejection: str = ""  # set by runner; emitted into next tick's prompt

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

        if self._last_rejection:
            lines.append(f"! ZONE REJECTED: {self._last_rejection}")
            self._last_rejection = ""

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
