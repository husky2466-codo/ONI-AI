# src/agent/protocol.py
import json
from dataclasses import dataclass
from typing import Any

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


@dataclass
class StateMessage:
    cycle: int
    time: float
    data: dict[str, Any]


def parse_state_message(raw: str) -> "StateMessage | None":
    """Parse a raw JSON line from the game. Returns StateMessage or None if not a state message."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if msg.get("type") != "state":
        return None

    data = msg.get("data", {})
    return StateMessage(
        cycle=int(data.get("cycle", 0)),
        time=float(data.get("time", 0.0)),
        data=data,
    )


def build_action(action: str, **kwargs: Any) -> dict[str, Any]:
    """Build an action command dict to send to the game."""
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown action: {action!r}. Valid: {VALID_ACTIONS}")
    return {"type": "action", "action": action, **kwargs}


def build_no_op() -> dict[str, Any]:
    """Build a no-op action command."""
    return build_action("no_op")


def build_place_perimeter(x1: int, y1: int, x2: int, y2: int, goal: str) -> dict[str, Any]:
    return build_action("place_perimeter", x1=x1, y1=y1, x2=x2, y2=y2, goal=goal)


def build_abandon_perimeter() -> dict[str, Any]:
    return build_action("abandon_perimeter")


def build_assign_research(tech_id: str) -> dict[str, Any]:
    return build_action("assign_research", tech_id=tech_id)


def build_enable_building(cell_x: int, cell_y: int, enabled: bool) -> dict[str, Any]:
    return build_action("enable_building", cell_x=cell_x, cell_y=cell_y, enabled=enabled)


def build_accept_print(offer_index: int) -> dict[str, Any]:
    return build_action("accept_print", offer_index=offer_index)
