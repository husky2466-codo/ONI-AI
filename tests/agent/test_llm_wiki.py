# tests/agent/test_llm_wiki.py
import os
import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from src.agent.llm import _format_state, _summarize_tiles, LLMAgent


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
    # New format: lists solid-cell coords, not element names
    assert "Solid tiles" in summary
    assert "(100,190)" in summary   # first solid cell
    assert "5" in summary           # 5 solid cells (Vacuum excluded)


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
    assert "(100,190)" in output   # solid cell coord listed


def test_format_state_no_tiles_section_when_absent():
    state = _make_state()
    output = _format_state(state)
    assert "Tile window" not in output


def _make_wiki_db(tmp_path: str) -> str:
    """Create a minimal wiki.db for testing."""
    db_path = os.path.join(tmp_path, "wiki.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE buildings (id TEXT PRIMARY KEY, name TEXT, body TEXT)")
    conn.execute("CREATE VIRTUAL TABLE buildings_fts USING fts5(name, body, content=buildings, content_rowid=rowid)")
    conn.execute("INSERT INTO buildings VALUES ('ManualGenerator','Manual Generator','Produces 400W. Needs a duplicant.')")
    conn.execute("INSERT INTO buildings_fts(rowid,name,body) SELECT rowid,name,body FROM buildings")
    conn.commit()
    conn.close()
    return db_path


def test_search_wiki_returns_results(tmp_path):
    db_path = _make_wiki_db(str(tmp_path))
    agent = LLMAgent.__new__(LLMAgent)
    agent._wiki_db = db_path
    result = agent._search_wiki("generator")
    assert "Manual Generator" in result
    assert "400W" in result


def test_search_wiki_no_db_returns_fallback():
    agent = LLMAgent.__new__(LLMAgent)
    agent._wiki_db = "/nonexistent/wiki.db"
    result = agent._search_wiki("anything")
    assert "not available" in result.lower()


def test_search_wiki_no_results(tmp_path):
    db_path = _make_wiki_db(str(tmp_path))
    agent = LLMAgent.__new__(LLMAgent)
    agent._wiki_db = db_path
    result = agent._search_wiki("xyzzy nonexistent query zzz")
    assert "No results" in result


# ---------------------------------------------------------------------------
# _format_state — spawned building tag + food-source survival check
# ---------------------------------------------------------------------------

def _make_building(btype: str, x: int = 100, y: int = 200, operational: bool = True) -> dict:
    return {"type": btype, "x": x, "y": y, "operational": operational}


def test_format_state_ration_box_suppresses_urgent_food_warning():
    """RationBox present but no MicrobeMusher → mild warning, not urgent."""
    state = _make_state(buildings=[_make_building("RationBox")])
    output = _format_state(state)
    assert "ration box" in output.lower()
    assert "no food source" not in output.lower()


def test_format_state_no_food_at_all_gives_urgent_warning():
    """No food source whatsoever → urgent warning."""
    state = _make_state(buildings=[])
    output = _format_state(state)
    assert "no food source" in output.lower()


def test_format_state_microbe_musher_suppresses_food_warning():
    """MicrobeMusher present → no food warning at all."""
    state = _make_state(buildings=[_make_building("MicrobeMusher")])
    output = _format_state(state)
    lines = output.split("\n")
    food_warning_lines = [l for l in lines if l.strip().startswith("!") and "food" in l.lower()]
    assert not food_warning_lines, f"Unexpected food warning: {food_warning_lines}"


def test_format_state_spawned_buildings_tagged():
    """Telepad and RationBox get [SPAWNED] tag; player-built Bed does not."""
    state = _make_state(buildings=[
        _make_building("Telepad", x=130, y=200),
        _make_building("RationBox", x=128, y=200),
        _make_building("Bed", x=116, y=201),
    ])
    output = _format_state(state)
    assert "Telepad [SPAWNED]" in output
    assert "RationBox [SPAWNED]" in output
    assert "Bed [SPAWNED]" not in output
