# Grid Vision + Wiki Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Gemini spatial awareness of the game map (tile element + mass in a window around the base) and the ability to query a local ONI wiki database before deciding each action.

**Architecture:** The C# mod serializes a tile window into the state JSON. A one-time Python scraper builds `data/wiki.db` (SQLite FTS5). `GeminiAgent` registers a `search_wiki` Gemini function declaration and runs a multi-turn loop so the model can look up game data before returning its action.

**Tech Stack:** C# / Newtonsoft.Json (mod), Python / sqlite3 / google-genai / requests + BeautifulSoup4 (scraper), pytest (tests)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `mod/ONIBridge/src/StateSerializer.cs` | Modify | Add `GetTiles()` method, add `tiles` field to `Serialize()` |
| `src/agent/llm.py` | Modify | Add `search_wiki` function declaration, multi-turn wiki loop, tile summary in `_format_state` |
| `src/agent/protocol.py` | Modify | No change to actions; `StateMessage` already carries `data` dict transparently |
| `scripts/build_wiki_db.py` | Create | One-time scraper: fetches ONI wiki pages, builds `data/wiki.db` with FTS5 |
| `tests/agent/test_llm_wiki.py` | Create | Unit tests for wiki search, tile summary formatting, action parsing with tool calls |
| `.gitignore` | Modify | Add `data/wiki.db` |

---

### Task 1: Add `tiles` field to StateSerializer (C#)

**Files:**
- Modify: `mod/ONIBridge/src/StateSerializer.cs`

- [ ] **Step 1: Add `GetTiles()` method to `StateSerializer`**

Open `mod/ONIBridge/src/StateSerializer.cs`. After the closing brace of `GetAlerts()` and before the final `}` of the class, add:

```csharp
private static object GetTiles()
{
    // Compute bounding box of all completed buildings
    int minX = int.MaxValue, maxX = int.MinValue;
    int minY = int.MaxValue, maxY = int.MinValue;
    bool hasBuildings = false;

    if (Components.BuildingCompletes != null)
    {
        foreach (BuildingComplete b in Components.BuildingCompletes)
        {
            if (b == null) continue;
            var pos = b.transform.position;
            int bx = (int)pos.x, by = (int)pos.y;
            if (bx < minX) minX = bx;
            if (bx > maxX) maxX = bx;
            if (by < minY) minY = by;
            if (by > maxY) maxY = by;
            hasBuildings = true;
        }
    }

    const int MARGIN = 15;
    int wx, wy;
    if (hasBuildings)
    {
        wx = minX - MARGIN;
        wy = minY - MARGIN;
    }
    else
    {
        // Fall back to 30×30 centered on world spawn
        wx = Grid.WidthInCells / 2 - 15;
        wy = Grid.HeightInCells / 2 - 15;
    }

    int ex = hasBuildings ? maxX + MARGIN : wx + 30;
    int ey = hasBuildings ? maxY + MARGIN : wy + 30;

    // Clamp to world bounds
    wx = System.Math.Max(0, wx);
    wy = System.Math.Max(0, wy);
    ex = System.Math.Min(Grid.WidthInCells - 1, ex);
    ey = System.Math.Min(Grid.HeightInCells - 1, ey);

    int w = ex - wx + 1;
    int h = ey - wy + 1;

    var data = new System.Collections.Generic.List<object>();
    for (int row = 0; row < h; row++)
    {
        for (int col = 0; col < w; col++)
        {
            int cx = wx + col;
            int cy = wy + row;
            int cell = Grid.XYToCell(cx, cy);
            if (!Grid.IsValidCell(cell))
            {
                data.Add(new object[] { "Invalid", 0f });
                continue;
            }
            string elementName = Grid.Element[cell]?.id.ToString() ?? "Vacuum";
            float mass = Grid.Mass[cell];
            data.Add(new object[] { elementName, System.Math.Round(mass, 1) });
        }
    }

    return new { x = wx, y = wy, w, h, data };
}
```

- [ ] **Step 2: Wire `GetTiles()` into `Serialize()`**

In the `Serialize()` method, add the `tiles` field alongside the existing fields:

```csharp
public static object Serialize()
{
    return new
    {
        cycle      = TryGet("cycle",      GetCycle,      0),
        time       = TryGet("time",       GetTime,       0f),
        resources  = TryGet("resources",  GetResources,  (object)new {}),
        duplicants = TryGet("duplicants", GetDuplicants, new List<object>()),
        buildings  = TryGet("buildings",  GetBuildings,  new List<object>()),
        alerts     = TryGet("alerts",     GetAlerts,     new List<string>()),
        tiles      = TryGet("tiles",      GetTiles,      (object)new {}),
    };
}
```

- [ ] **Step 3: Rebuild the mod and deploy**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge
dotnet build
```

Expected: Build succeeded, 0 errors.

Copy the resulting DLL to the ONI mods folder on the Windows PC (same process as previous mod deployments). Reload the game and verify no `[ONIBridge] tiles failed:` warnings appear in the game log.

- [ ] **Step 4: Commit**

```bash
git add mod/ONIBridge/src/StateSerializer.cs
git commit -m "feat: serialize tile window (element + mass) in state payload"
```

---

### Task 2: Tile summary in Python prompt formatter

**Files:**
- Modify: `src/agent/llm.py`
- Test: `tests/agent/test_llm_wiki.py`

- [ ] **Step 1: Write failing test for tile summary**

Create `tests/agent/test_llm_wiki.py`:

```python
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
    assert "3x100" in summary or "Sandstone" in summary
    assert "Oxygen" in summary
    assert "6 tiles" in summary or "6" in summary


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
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/agent/test_llm_wiki.py -v
```

Expected: FAIL — `_summarize_tiles` not defined.

- [ ] **Step 3: Implement `_summarize_tiles` in `llm.py`**

Add this function to `src/agent/llm.py` after `_format_state`:

```python
def _summarize_tiles(tiles: dict) -> "str | None":
    """Return a compact text summary of a tile window for inclusion in the prompt."""
    data = tiles.get("data")
    if not data:
        return None
    x, y, w, h = tiles.get("x", 0), tiles.get("y", 0), tiles.get("w", 0), tiles.get("h", 0)
    total = len(data)
    counts: dict[str, int] = {}
    for entry in data:
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            name = str(entry[0])
            counts[name] = counts.get(name, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:6]
    top_str = ", ".join(f"{n}({c})" for n, c in top)
    return f"Tile window: x={x} y={y} w={w} h={h} ({total} tiles) | top elements: {top_str}"
```

- [ ] **Step 4: Add tile summary to `_format_state`**

Inside `_format_state`, after the MISSING survival buildings block, add:

```python
    tile_summary = _summarize_tiles(data.get("tiles", {}))
    if tile_summary:
        lines.append("")
        lines.append(tile_summary)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/agent/test_llm_wiki.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agent/llm.py tests/agent/test_llm_wiki.py
git commit -m "feat: add tile window summary to Gemini prompt"
```

---

### Task 3: Build the ONI wiki SQLite database

**Files:**
- Create: `scripts/build_wiki_db.py`
- Modify: `.gitignore`

- [ ] **Step 1: Install scraping dependency**

```bash
pip install requests beautifulsoup4
```

Expected: Successfully installed (or already satisfied).

- [ ] **Step 2: Add `data/wiki.db` to `.gitignore`**

Open `.gitignore` and add:

```
data/wiki.db
data/*.db
```

- [ ] **Step 3: Create `scripts/build_wiki_db.py`**

```python
#!/usr/bin/env python3
"""
One-time script: scrape the ONI wiki and build data/wiki.db (SQLite FTS5).

Usage:
    python3 scripts/build_wiki_db.py

Requires: requests, beautifulsoup4
    pip install requests beautifulsoup4
"""
from __future__ import annotations

import sqlite3
import time
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

WIKI_BASE = "https://oxygennotincluded.wiki.gg"
DB_PATH = Path("data/wiki.db")

PAGES: dict[str, list[str]] = {
    "buildings": [
        "/wiki/Manual_Generator", "/wiki/Battery", "/wiki/Wire",
        "/wiki/Oxygen_Diffuser", "/wiki/Electrolyzer", "/wiki/Algae_Terrarium",
        "/wiki/Outhouse", "/wiki/Lavatory", "/wiki/Wash_Basin",
        "/wiki/Cot", "/wiki/Microbe_Musher", "/wiki/Electric_Grill",
        "/wiki/Mealwood", "/wiki/Farm_Tile", "/wiki/Planter_Box",
        "/wiki/Research_Station", "/wiki/Air_Deodorizer", "/wiki/Carbon_Skimmer",
        "/wiki/Pitcher_Pump", "/wiki/Water_Sieve",
    ],
    "elements": [
        "/wiki/Oxygen", "/wiki/Carbon_Dioxide", "/wiki/Hydrogen",
        "/wiki/Water", "/wiki/Polluted_Water", "/wiki/Dirt",
        "/wiki/Sandstone", "/wiki/Algae", "/wiki/Coal",
    ],
    "foods": [
        "/wiki/Mush_Bar", "/wiki/Pickled_Meal", "/wiki/Meal_Lice",
        "/wiki/Grilled_Liceloaf", "/wiki/BBQ",
    ],
    "research": [
        "/wiki/Farming_Tech", "/wiki/Meal_Preparation",
        "/wiki/Sanitation", "/wiki/Jobs", "/wiki/Interior_Decor",
        "/wiki/Basic_Farming", "/wiki/Plumbing",
    ],
}

HEADERS = {"User-Agent": "ONI-AI-WikiScraper/1.0 (research bot)"}


def fetch_page_text(path: str) -> tuple[str, str]:
    """Return (title, body_text) for a wiki page path."""
    url = WIKI_BASE + path
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_el = soup.find("h1", id="firstHeading") or soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else path.split("/")[-1]

    content = soup.find("div", id="mw-content-text") or soup.find("div", class_="mw-parser-output")
    if not content:
        return title, ""

    # Remove infobox tables that are noisy
    for tag in content.find_all(["table", "sup", "span"]):
        tag.decompose()

    text = content.get_text(separator=" ", strip=True)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return title, text[:2000]  # cap at 2000 chars per page


def create_schema(conn: sqlite3.Connection) -> None:
    for table in ("buildings", "elements", "foods", "research"):
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id   TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                body TEXT NOT NULL
            )
        """)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {table}_fts
            USING fts5(name, body, content={table}, content_rowid=rowid)
        """)
    conn.commit()


def insert_row(conn: sqlite3.Connection, table: str, page_id: str, name: str, body: str) -> None:
    conn.execute(
        f"INSERT OR REPLACE INTO {table} (id, name, body) VALUES (?, ?, ?)",
        (page_id, name, body)
    )
    conn.execute(
        f"INSERT OR REPLACE INTO {table}_fts(rowid, name, body) "
        f"SELECT rowid, name, body FROM {table} WHERE id = ?",
        (page_id,)
    )


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    total = sum(len(v) for v in PAGES.values())
    done = 0
    errors = 0

    for table, paths in PAGES.items():
        for path in paths:
            page_id = path.split("/")[-1]
            try:
                name, body = fetch_page_text(path)
                insert_row(conn, table, page_id, name, body)
                conn.commit()
                done += 1
                print(f"[{done}/{total}] {table}/{page_id} — {len(body)} chars")
            except Exception as e:
                errors += 1
                print(f"  ERROR {path}: {e}", file=sys.stderr)
            time.sleep(0.5)  # be polite to the wiki server

    conn.close()
    print(f"\nDone. {done} pages, {errors} errors. DB: {DB_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the scraper**

```bash
python3 scripts/build_wiki_db.py
```

Expected: Output shows pages being fetched, ends with `Done. N pages, 0 errors. DB: data/wiki.db`. A non-zero error count is acceptable (some wiki pages may redirect or 404).

- [ ] **Step 5: Verify the database**

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data/wiki.db')
for t in ('buildings','elements','foods','research'):
    n = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t}: {n} rows')
rows = conn.execute(\"SELECT name, body FROM buildings_fts WHERE buildings_fts MATCH 'generator' LIMIT 2\").fetchall()
for r in rows: print(r[0], ':', r[1][:80])
conn.close()
"
```

Expected: All tables show at least 1 row; FTS query returns results for 'generator'.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_wiki_db.py .gitignore
git commit -m "feat: add ONI wiki scraper and build data/wiki.db (gitignored)"
```

---

### Task 4: Add wiki tool calling to GeminiAgent

**Files:**
- Modify: `src/agent/llm.py`
- Test: `tests/agent/test_llm_wiki.py`

- [ ] **Step 1: Write failing tests for wiki search and tool loop**

Add to `tests/agent/test_llm_wiki.py`:

```python
import sqlite3
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.agent.llm import GeminiAgent


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
    agent = GeminiAgent.__new__(GeminiAgent)
    agent._wiki_db = db_path
    result = agent._search_wiki("generator")
    assert "Manual Generator" in result
    assert "400W" in result


def test_search_wiki_no_db_returns_fallback():
    agent = GeminiAgent.__new__(GeminiAgent)
    agent._wiki_db = "/nonexistent/wiki.db"
    result = agent._search_wiki("anything")
    assert "not available" in result.lower()


def test_search_wiki_no_results():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _make_wiki_db(tmp)
        agent = GeminiAgent.__new__(GeminiAgent)
        agent._wiki_db = db_path
        result = agent._search_wiki("xyzzy nonexistent query zzz")
        assert "No results" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/agent/test_llm_wiki.py::test_search_wiki_returns_results tests/agent/test_llm_wiki.py::test_search_wiki_no_db_returns_fallback tests/agent/test_llm_wiki.py::test_search_wiki_no_results -v
```

Expected: FAIL — `GeminiAgent` has no `_search_wiki` or `_wiki_db` attribute.

- [ ] **Step 3: Add `_wiki_db` and `_search_wiki` to `GeminiAgent`**

In `src/agent/llm.py`, modify `GeminiAgent.__init__` to add:

```python
import sqlite3 as _sqlite3

# Inside __init__, after existing self.total_cost_usd = 0.0:
self._wiki_db = "data/wiki.db"
```

Add this method to `GeminiAgent` after `_parse_action`:

```python
def _search_wiki(self, query: str) -> str:
    """Query the local ONI wiki SQLite database. Returns top 3 results."""
    if not Path(self._wiki_db).exists():
        return "Wiki database not available."
    try:
        conn = _sqlite3.connect(self._wiki_db)
        results: list[str] = []
        for table in ("buildings", "elements", "foods", "research"):
            try:
                rows = conn.execute(
                    f"SELECT name, body FROM {table}_fts WHERE {table}_fts MATCH ? LIMIT 2",
                    (query,)
                ).fetchall()
                for name, body in rows:
                    results.append(f"{name}: {body[:300]}")
            except _sqlite3.OperationalError:
                # Table may not exist if scraper only partially ran
                continue
        conn.close()
        if not results:
            return "No results found."
        return "\n\n".join(results[:3])
    except Exception as e:
        logger.warning("Wiki search failed: %s", e)
        return "Wiki search error."
```

Also add the `Path` import at the top of `llm.py` if not already present:
```python
from pathlib import Path
```

- [ ] **Step 4: Run wiki search tests to confirm they pass**

```bash
pytest tests/agent/test_llm_wiki.py::test_search_wiki_returns_results tests/agent/test_llm_wiki.py::test_search_wiki_no_db_returns_fallback tests/agent/test_llm_wiki.py::test_search_wiki_no_results -v
```

Expected: All 3 PASS.

- [ ] **Step 5: Implement the multi-turn tool calling loop in `decide()`**

Replace the `decide()` method in `GeminiAgent` with:

```python
def decide(self, state_data: dict[str, Any]) -> dict[str, Any]:
    """
    Given a state snapshot dict, return an ActionCommand dict.
    Gemini may call search_wiki() up to 2 times before returning its action.
    Falls back to no_op on any failure.
    """
    from google.genai import types as _types

    search_wiki_fn = _types.FunctionDeclaration(
        name="search_wiki",
        description=(
            "Search the ONI wiki for game data: building stats, element properties, "
            "food recipes, or research costs. Use this when you need to look up "
            "a building's power draw, inputs, outputs, or size before placing it."
        ),
        parameters=_types.Schema(
            type=_types.Type.OBJECT,
            properties={"query": _types.Schema(type=_types.Type.STRING)},
            required=["query"],
        ),
    )
    tool = _types.Tool(function_declarations=[search_wiki_fn])

    prompt = _format_state(state_data)
    contents = [prompt]
    max_wiki_calls = 2
    wiki_calls = 0

    try:
        while True:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=1024,
                    thinking_config=_types.ThinkingConfig(thinking_budget=0),
                    tools=[tool],
                ),
            )

            # Track token usage and cost
            usage = response.usage_metadata
            if usage:
                inp = usage.prompt_token_count or 0
                out = usage.candidates_token_count or 0
                self.total_input_tokens  += inp
                self.total_output_tokens += out
                self.total_calls         += 1
                self.total_cost_usd += (inp / 1_000_000) * _COST_INPUT_PER_M
                self.total_cost_usd += (out / 1_000_000) * _COST_OUTPUT_PER_M

            # Check for function call
            candidate = response.candidates[0] if response.candidates else None
            fn_call = None
            if candidate and candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fn_call = part.function_call
                        break

            if fn_call and fn_call.name == "search_wiki" and wiki_calls < max_wiki_calls:
                query = fn_call.args.get("query", "")
                logger.info("  -> Gemini wiki call: %r", query)
                wiki_result = self._search_wiki(query)
                wiki_calls += 1

                # Append assistant turn + function response to contents
                contents.append(candidate.content)
                contents.append(_types.Content(parts=[
                    _types.Part(function_response=_types.FunctionResponse(
                        name="search_wiki",
                        response={"result": wiki_result},
                    ))
                ]))
                continue  # loop again with wiki result in context

            # No function call — parse the action text
            raw = response.text.strip() if response.text else ""
            logger.debug("Gemini raw response: %s", raw)
            return self._parse_action(raw)

    except Exception as e:
        logger.warning("Gemini call failed: %s — sending no_op", e)
        return build_no_op()
```

- [ ] **Step 6: Run the full test suite**

```bash
pytest tests/agent/test_llm_wiki.py -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agent/llm.py tests/agent/test_llm_wiki.py
git commit -m "feat: add wiki tool calling to GeminiAgent with multi-turn loop"
```

---

### Task 5: End-to-end smoke test with live game

**Files:**
- No code changes — verification only

- [ ] **Step 1: Run the wiki scraper if not already done**

```bash
python3 scripts/build_wiki_db.py
```

Confirm `data/wiki.db` exists with content (see Task 3 Step 5 verification command).

- [ ] **Step 2: Start the dashboard and runner**

```bash
GOOGLE_API_KEY=<your_key> python3 examples/dashboard/server.py
```

Open http://localhost:8181 in the browser. Click Start.

- [ ] **Step 3: Watch runner logs for wiki calls**

In the runner log, you should see lines like:

```
  -> Gemini wiki call: 'Manual Generator power output'
  -> AI action: {'action': 'place_building', 'building_id': 'ManualGenerator', ...}
```

Also verify tile data appears in the log — look for `[tick N | cycle M]` lines with non-zero O2/food values after the game has loaded.

- [ ] **Step 4: Verify no regressions**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "feat: grid vision + wiki tool calling — end-to-end verified"
```
