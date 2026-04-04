# tests/environments/live_oni/test_action_adapter.py
import pytest
from src.environments.live_oni.action_adapter import action_to_bridge, BUILDING_TYPE_TO_PREFAB
from src.environments.mini_oni.actions import (
    generate_action_space, NoOpAction, DigAction, PlaceBuildingAction,
    PriorityAction, DuplicantAction, ActionType, Region, TaskType, SkillType,
)
from src.environments.mini_oni.building_types import BuildingType

# generate_action_space(64, 64) hits MAX_ACTIONS=200 and keeps only NoOp +
# PlaceBuildingAction due to the trimming logic.  Build a representative
# action space that contains one instance of every action type so we can
# exercise each branch of action_to_bridge().
ACTION_SPACE: list = [
    NoOpAction(),
    PlaceBuildingAction(BuildingType.OXYGEN_DIFFUSER, Region(0, 0, 0, 0)),
    DigAction(Region(2, 3, 2, 3)),
    PriorityAction(TaskType.BUILD, 5),
    DuplicantAction(0, SkillType.MINING),
]


def _find_action_idx(action_space, action_type):
    """Find first action index of given type."""
    for i, a in enumerate(action_space):
        if isinstance(a, action_type):
            return i
    raise ValueError(f"No action of type {action_type}")


def test_no_op_action():
    idx = _find_action_idx(ACTION_SPACE, NoOpAction)
    result = action_to_bridge(idx, ACTION_SPACE)
    assert result == {"type": "action", "action": "no_op"}


def test_dig_action_has_cell():
    idx = _find_action_idx(ACTION_SPACE, DigAction)
    result = action_to_bridge(idx, ACTION_SPACE)
    assert result["action"] == "dig"
    assert "cell_x" in result
    assert "cell_y" in result
    assert isinstance(result["cell_x"], int)
    assert isinstance(result["cell_y"], int)
    assert result["cell_x"] == 2
    assert result["cell_y"] == 3


def test_place_building_has_building_id():
    idx = _find_action_idx(ACTION_SPACE, PlaceBuildingAction)
    result = action_to_bridge(idx, ACTION_SPACE)
    assert result["action"] == "place_building"
    assert "building_id" in result
    assert "cell_x" in result
    assert "cell_y" in result
    assert result["building_id"] == "OxygenDiffuser"


def test_set_priority_action():
    idx = _find_action_idx(ACTION_SPACE, PriorityAction)
    result = action_to_bridge(idx, ACTION_SPACE)
    assert result["action"] == "set_priority"
    assert "cell_x" in result
    assert "cell_y" in result
    assert 1 <= result["priority"] <= 9


def test_duplicant_action_maps_to_no_op():
    # DuplicantAction has no bridge equivalent — must fall back to no_op
    idx = _find_action_idx(ACTION_SPACE, DuplicantAction)
    result = action_to_bridge(idx, ACTION_SPACE)
    assert result["action"] == "no_op"


def test_invalid_index_maps_to_no_op():
    result = action_to_bridge(999999, ACTION_SPACE)
    assert result["action"] == "no_op"


def test_negative_index_maps_to_no_op():
    result = action_to_bridge(-1, ACTION_SPACE)
    assert result["action"] == "no_op"


def test_building_type_to_prefab_covers_essential_buildings():
    from src.environments.mini_oni.building_types import ESSENTIAL_BUILDINGS
    for bt in ESSENTIAL_BUILDINGS:
        assert bt.name in BUILDING_TYPE_TO_PREFAB, f"Missing prefab for {bt.name}"
