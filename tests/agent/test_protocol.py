# tests/agent/test_protocol.py
import json
import pytest
from src.agent.protocol import (
    parse_state_message,
    build_action,
    build_no_op,
    StateMessage,
)


def test_parse_valid_state_message():
    raw = json.dumps({
        "type": "state",
        "data": {
            "cycle": 3,
            "time": 1.5,
            "resources": {"oxygen_kg": 12.5, "water_kg": 80.0, "food_kcal_today": 2000.0, "power_kw": 1.2, "co2_kg": 0.3},
            "duplicants": [{"id": 1, "name": "Higby", "x": 10, "y": 8, "stress": 0.1, "health": 100.0, "current_task": "dig"}],
            "buildings": [{"type": "OxygenDiffuser", "x": 5, "y": 5, "operational": True}],
            "alerts": [],
        }
    })
    msg = parse_state_message(raw)
    assert isinstance(msg, StateMessage)
    assert msg.cycle == 3
    assert msg.data["resources"]["oxygen_kg"] == 12.5
    assert msg.data["duplicants"][0]["name"] == "Higby"


def test_parse_ignores_non_state_type():
    raw = json.dumps({"type": "ack", "action": "no_op", "success": True})
    msg = parse_state_message(raw)
    assert msg is None


def test_parse_invalid_json_raises():
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_state_message("not json {{{")


def test_build_no_op():
    action = build_no_op()
    assert action["type"] == "action"
    assert action["action"] == "no_op"


def test_build_place_building():
    action = build_action("place_building", building_id="OxygenDiffuser", cell_x=12, cell_y=8)
    assert action["action"] == "place_building"
    assert action["building_id"] == "OxygenDiffuser"
    assert action["cell_x"] == 12
    assert action["cell_y"] == 8


def test_build_dig():
    action = build_action("dig", cell_x=5, cell_y=3)
    assert action["action"] == "dig"
    assert action["cell_x"] == 5
    assert action["cell_y"] == 3


def test_build_set_priority():
    action = build_action("set_priority", cell_x=5, cell_y=3, priority=7)
    assert action["action"] == "set_priority"
    assert action["priority"] == 7


def test_build_action_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown action"):
        build_action("fly_rocket")


def test_build_set_speed_action():
    action = build_action("set_speed", speed=2)
    assert action["action"] == "set_speed"
    assert action["speed"] == 2


def test_set_speed_is_valid_action():
    from src.agent.protocol import VALID_ACTIONS
    assert "set_speed" in VALID_ACTIONS


def test_build_unknown_action_raises():
    with pytest.raises(ValueError):
        build_action("invalid_action")
