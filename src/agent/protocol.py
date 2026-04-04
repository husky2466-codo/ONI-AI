# src/agent/protocol.py
import json
from dataclasses import dataclass
from typing import Any

VALID_ACTIONS = {"place_building", "dig", "cancel_dig", "set_priority", "no_op", "set_speed"}


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
