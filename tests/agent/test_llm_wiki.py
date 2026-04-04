# tests/agent/test_llm_wiki.py
import pytest
from src.agent.llm import _format_state, _summarize_tiles


def _make_state(**overrides):
    base = {
        "cycle": 1,
        "time": 0.0,
        "resources": {
            "oxygen_kg": 10.0, "water_kg": 50.0,
            "food_kcal": 5000.0, "power_kw": 0.0, "co2_kg": 0.0,
        },
        "duplicants": [],
        "buildings": [],
        "alerts": [],
    }
    base.update(overrides)
    return base


def test_summarize_tiles_counts_elements():
    tiles = {
        "x": 100, "y": 190, "w": 3, "h": 2,
        "data": [
            ["Sandstone", 1800.0], ["Vacuum", 0.0], ["Sandstone", 900.0],
            ["Oxygen", 450.0],     ["Dirt", 600.0],  ["Sandstone", 1200.0],
        ]
    }
    summary = _summarize_tiles(tiles)
    assert "Sandstone" in summary
    assert "Oxygen" in summary
    assert "6" in summary


def test_summarize_tiles_missing_returns_none():
    result = _summarize_tiles({})
    assert result is None


def test_format_state_includes_tile_summary():
    tiles = {
        "x": 100, "y": 190, "w": 2, "h": 1,
        "data": [["Sandstone", 1800.0], ["Vacuum", 0.0]],
    }
    state = _make_state(tiles=tiles)
    output = _format_state(state)
    assert "Tile window" in output
    assert "Sandstone" in output


def test_format_state_no_tiles_section_when_absent():
    state = _make_state()
    output = _format_state(state)
    assert "Tile window" not in output
