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
