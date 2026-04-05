# tests/agent/test_runner_validation.py
"""Tests for coordinate validation helpers in runner.py."""
import pytest

from src.agent.runner import _build_solid_cells, _validate_dig


def _tiles(data: list, x: int = 100, y: int = 190, w: int = 3, h: int = 2) -> dict:
    return {"x": x, "y": y, "w": w, "h": h, "data": data}


# ---------------------------------------------------------------------------
# _build_solid_cells
# ---------------------------------------------------------------------------

def test_build_solid_cells_empty():
    assert _build_solid_cells({}) == set()


def test_build_solid_cells_all_open():
    state = {"tiles": _tiles([["Vacuum", 0.0]] * 6)}
    assert _build_solid_cells(state) == set()


def test_build_solid_cells_mixed():
    # 3x2 grid: only (100,190) and (102,190) are solid
    data = [
        ["Sandstone", 1800.0], ["Vacuum", 0.0], ["Sandstone", 900.0],
        ["Oxygen", 0.0],       ["Vacuum", 0.0], ["Vacuum", 0.0],
    ]
    state = {"tiles": _tiles(data)}
    solid = _build_solid_cells(state)
    assert (100, 190) in solid
    assert (102, 190) in solid
    assert (101, 190) not in solid
    assert (100, 191) not in solid  # Oxygen mass=0 → not solid


# ---------------------------------------------------------------------------
# _validate_dig
# ---------------------------------------------------------------------------

def _solid_state() -> dict:
    """State where (101,190) is solid, (100,190) and (102,190) are open."""
    data = [
        ["Vacuum", 0.0], ["Sandstone", 1800.0], ["Vacuum", 0.0],
        ["Vacuum", 0.0], ["Vacuum", 0.0],        ["Vacuum", 0.0],
    ]
    return {"tiles": _tiles(data)}


def test_validate_dig_solid_cell_passes_through():
    action = {"type": "action", "action": "dig", "cell_x": 101, "cell_y": 190}
    result = _validate_dig(action, _solid_state())
    assert result["action"] == "dig"
    assert result["cell_x"] == 101


def test_validate_dig_open_cell_returns_no_op():
    action = {"type": "action", "action": "dig", "cell_x": 100, "cell_y": 190}
    result = _validate_dig(action, _solid_state())
    assert result["action"] == "no_op"


def test_validate_dig_outside_window_passes_through():
    action = {"type": "action", "action": "dig", "cell_x": 50, "cell_y": 50}
    result = _validate_dig(action, _solid_state())
    assert result["action"] == "dig"  # can't confirm, so let it through


def test_validate_dig_missing_coords_passes_through():
    action = {"type": "action", "action": "dig"}
    result = _validate_dig(action, _solid_state())
    assert result["action"] == "dig"
