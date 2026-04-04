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
    next_tasks = [
        f"Place {s['type']} at ({s['abs_x']},{s['abs_y']})"
        for s in ordered[:5]
    ]

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
# ActivePerimeter
# ---------------------------------------------------------------------------

class ActivePerimeter:
    def __init__(
        self,
        id: str,
        goal: str,
        bounds: dict,
        blueprint_id: str,
        cycle_started: int,
    ) -> None:
        self.id = id
        self.goal = goal
        self.bounds = bounds
        self.blueprint_id = blueprint_id
        self.cycle_started = cycle_started
        self.task_board: TaskBoard | None = None

    @classmethod
    def from_state(
        cls,
        perimeter_payload: dict,
        cycle: int,
        blueprint_id: str = "",
    ) -> "ActivePerimeter":
        return cls(
            id=perimeter_payload["id"],
            goal=perimeter_payload["goal"],
            bounds=perimeter_payload["bounds"],
            blueprint_id=blueprint_id,
            cycle_started=cycle,
        )

    def update(self, state: dict, blueprint: dict | None, storage: list[dict]) -> None:
        if blueprint:
            self.task_board = _build_task_board(blueprint, state, self.bounds, storage)


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
    def __init__(self, blueprint_library: BlueprintLibrary | None = None) -> None:
        self.active: ActivePerimeter | None = None
        self.history: list[LedgerEntry] = []
        self._library = blueprint_library or BlueprintLibrary()
        self._autocomplete_pending: bool = False  # signals runner to send abandon_perimeter

    @property
    def autocomplete_pending(self) -> bool:
        """True when the task board hit 100% and runner should send abandon_perimeter."""
        return self._autocomplete_pending

    def clear_autocomplete(self) -> None:
        self._autocomplete_pending = False

    def on_state(self, state: dict) -> None:
        """Call each tick with the parsed state dict. Handles perimeter lifecycle."""
        p = state.get("perimeter")
        storage = state.get("storage", [])

        if p is None:
            # Mod has no active perimeter — clear our active too
            self.active = None
            return

        if self.active is None:
            # New perimeter appeared — look up blueprint
            bounds = p["bounds"]
            w = bounds["x2"] - bounds["x1"]
            h = bounds["y2"] - bounds["y1"]
            blueprint = self._library.select(p["goal"], w, h)
            blueprint_id = blueprint["id"] if blueprint else ""
            self.active = ActivePerimeter.from_state(p, state.get("cycle", 0), blueprint_id)
            logger.info(
                "Perimeter activated: %s goal=%s blueprint=%s",
                self.active.id, self.active.goal, blueprint_id,
            )

        # Update task board
        blueprint = self._library.get(self.active.blueprint_id) if self.active.blueprint_id else None
        self.active.update(state, blueprint, storage)

        # Auto-complete check
        if self.active.task_board and self.active.task_board.pct >= 100.0:
            logger.info("Perimeter %s auto-complete (100%%) — signalling runner", self.active.id)
            self._archive("complete", state.get("cycle", 0))
            self._autocomplete_pending = True

    def on_abandon(self, cycle: int) -> None:
        """Call when the runner explicitly sends abandon_perimeter."""
        self._archive("abandoned", cycle)

    def _archive(self, status: str, cycle: int) -> None:
        if self.active:
            self.history.append(LedgerEntry(
                id=self.active.id,
                goal=self.active.goal,
                blueprint_id=self.active.blueprint_id,
                bounds=self.active.bounds,
                status=status,
                cycle_started=self.active.cycle_started,
                cycle_ended=cycle,
            ))
            self.active = None

    def format_context(self) -> str:
        """Compact string for injection into the AI prompt."""
        lines: list[str] = []

        if self.active:
            b = self.active.bounds
            lines.append(
                f"ACTIVE PERIMETER: {self.active.goal} @ "
                f"({b['x1']},{b['y1']})-({b['x2']},{b['y2']})"
            )
            tb = self.active.task_board
            if tb:
                lines.append(f"  Progress: {tb.completed}/{tb.total} steps ({tb.pct:.0f}%)")
                for t in tb.next_tasks[:3]:
                    lines.append(f"  Next: {t}")
                for pr in tb.prerequisites[:3]:
                    lines.append(f"  Needs: {pr}")
            else:
                lines.append("  (task board loading...)")
        else:
            lines.append("No active perimeter — use place_perimeter to declare next build zone.")

        if self.history:
            recent = [e.goal for e in self.history[-3:]]
            lines.append(f"Completed zones: {len(self.history)} ({', '.join(recent)})")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot for episode logging."""
        return {
            "active": {
                "id": self.active.id,
                "goal": self.active.goal,
                "bounds": self.active.bounds,
                "blueprint_id": self.active.blueprint_id,
                "cycle_started": self.active.cycle_started,
                "task_board": {
                    "total": self.active.task_board.total,
                    "completed": self.active.task_board.completed,
                    "pct": self.active.task_board.pct,
                } if self.active.task_board else None,
            } if self.active else None,
            "history": [
                {
                    "id": e.id,
                    "goal": e.goal,
                    "status": e.status,
                    "cycle_started": e.cycle_started,
                    "cycle_ended": e.cycle_ended,
                }
                for e in self.history
            ],
        }
