"""
Action adapter for LiveONI environment.

Maps gym-style integer action indices to bridge action dicts
that the C# TCP server understands.
"""
from __future__ import annotations

from typing import Any

from src.environments.mini_oni.actions import (
    Action,
    DigAction,
    DuplicantAction,
    NoOpAction,
    PlaceBuildingAction,
    PriorityAction,
)

# Mapping from BuildingType.name → prefab string used by the C# bridge.
# Must cover every key in ESSENTIAL_BUILDINGS.
BUILDING_TYPE_TO_PREFAB: dict[str, str] = {
    "OXYGEN_DIFFUSER":  "OxygenDiffuser",
    "ELECTROLYZER":     "Electrolyzer",
    "LADDER":           "Ladder",
    "TILE":             "Tile",
    "INSULATED_TILE":   "InsulatedTile",
    "OUTHOUSE":         "Outhouse",
    "WASH_BASIN":       "WashBasin",
    "COT":              "Bed",
    "MEALWOOD":         "Mealwood",
    "MICROBE_MUSHER":   "MicrobeMusher",
    "MANUAL_GENERATOR": "ManualGenerator",
    "BATTERY":          "Battery",
    "WATER_SIEVE":      "WaterSieve",
    "LIQUID_PUMP":      "LiquidPumpingStation",
    "GAS_PUMP":         "GasPumpingStation",
}

_NO_OP_DICT: dict[str, Any] = {"type": "action", "action": "no_op"}


def action_to_bridge(action_idx: int, action_space: list[Action]) -> dict[str, Any]:
    """Convert an action index to a bridge action dict.

    Args:
        action_idx: Integer index into *action_space*.
        action_space: List of :class:`Action` objects produced by
            ``generate_action_space()``.

    Returns:
        A ``dict`` understood by the C# TCP bridge, always containing at
        minimum ``{"type": "action", "action": <str>}``.
        Out-of-range indices and :class:`DuplicantAction` fall back to
        ``no_op``.
    """
    if action_idx < 0 or action_idx >= len(action_space):
        return _NO_OP_DICT.copy()

    action = action_space[action_idx]

    # --- no-op -----------------------------------------------------------
    if isinstance(action, NoOpAction):
        return {"type": "action", "action": "no_op"}

    # --- dig -------------------------------------------------------------
    if isinstance(action, DigAction):
        return {
            "type":   "action",
            "action": "dig",
            "cell_x": int(action.region.x1),
            "cell_y": int(action.region.y1),
        }

    # --- place building --------------------------------------------------
    if isinstance(action, PlaceBuildingAction):
        prefab = BUILDING_TYPE_TO_PREFAB.get(action.building_type.name, "Unknown")
        return {
            "type":        "action",
            "action":      "place_building",
            "building_id": prefab,
            "cell_x":      int(action.region.x1),
            "cell_y":      int(action.region.y1),
        }

    # --- set priority ----------------------------------------------------
    if isinstance(action, PriorityAction):
        priority = max(1, min(9, int(action.priority_level)))
        return {
            "type":     "action",
            "action":   "set_priority",
            "cell_x":   0,
            "cell_y":   0,
            "priority": priority,
        }

    # --- duplicant assignment (no bridge equivalent) ---------------------
    if isinstance(action, DuplicantAction):
        return {"type": "action", "action": "no_op"}

    # fallback for any unknown subclass
    return {"type": "action", "action": "no_op"}
