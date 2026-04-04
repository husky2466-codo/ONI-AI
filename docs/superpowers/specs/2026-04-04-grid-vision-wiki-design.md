# Grid Vision + Wiki Tool Calling — Design Spec

**Date:** 2026-04-04
**Status:** Approved

---

## Overview

Two related AI capability enhancements:

1. **Grid Vision** — The C# ONIBridge mod serializes tile-level map data (element + mass) in a window around the active base and includes it in every state message. Gemini can then reason about what tiles exist, what's been dug, and where to place buildings.

2. **Wiki Tool Calling** — Gemini is given a `search_wiki(query)` function it can call at decision time to look up game data (building stats, element properties, recipes) from a local SQLite database built by scraping the ONI wiki.

---

## Spec A1: Grid Serialization (C# Mod)

### Location
`mod/ONIBridge/src/StateSerializer.cs`

### Behavior

A new `GetTiles()` method is added to `StateSerializer`. It is called from `Serialize()` wrapped in `TryGet` like all other sections:

```csharp
tiles = TryGet("tiles", GetTiles, (object)new {}),
```

### Window Calculation

1. Iterate all `BuildingComplete` positions to find `minX`, `maxX`, `minY`, `maxY`.
2. If no buildings exist, fall back to a 30×30 window centered on world spawn point (`Grid.WidthInCells / 2`, `Grid.HeightInCells / 2`).
3. Expand the bounding box by 15 tiles in each direction.
4. Clamp to world bounds: `[0, Grid.WidthInCells)` and `[0, Grid.HeightInCells)`.

### Per-Cell Data

For each cell in the window, read:
- `Grid.Element[cell].id.ToString()` — element name (e.g., `"Sandstone"`, `"Oxygen"`, `"Vacuum"`)
- `Grid.Mass[cell]` — mass in grams (float)

### Output Shape

```json
"tiles": {
  "x": 105,
  "y": 183,
  "w": 62,
  "h": 32,
  "data": [["Sandstone", 1800.0], ["Vacuum", 0.0], ["Oxygen", 450.2], ...]
}
```

- `x`, `y` — top-left corner of the window (world coordinates)
- `w`, `h` — width and height of the window in tiles
- `data` — flat row-major array of `[element_name, mass_g]` pairs, length `w * h`
- Row 0 is the bottom row (y = y_origin), row h-1 is the top row. Column 0 is leftmost (x = x_origin).

### Error Handling

If `Grid.Element` or `Grid.Mass` throw (world not loaded, out of bounds), `TryGet` catches and returns an empty object `{}`. The Python side treats a missing or empty `tiles` field as "no grid data yet."

### Performance Note

A 60×60 window = 3600 cells. Each cell serializes as a 2-element JSON array. This adds roughly 80–120 KB to the state payload. Acceptable given the 1-second tick interval and LAN connection.

---

## Spec A2: Wiki Database (Python)

### Location
- Scraper: `scripts/build_wiki_db.py`
- Database: `data/wiki.db` (SQLite, FTS5, gitignored)

### Scraper

A one-time script that fetches pages from the ONI wiki (wiki.playonigame.com) and populates `data/wiki.db`. Tables:

| Table | Columns | Content |
|-------|---------|---------|
| `buildings` | `id`, `name`, `body` | Building stats: power, inputs, outputs, size, description |
| `elements` | `id`, `name`, `body` | Element properties: state, thermal conductivity, specific heat |
| `foods` | `id`, `name`, `body` | Food items: calories, ingredients, difficulty |
| `research` | `id`, `name`, `body` | Research nodes: cost, unlocks |

Each table has an FTS5 virtual table for full-text search.

The scraper is a standalone script. It does not need to run at agent startup — run it once manually: `python3 scripts/build_wiki_db.py`.

### Database Schema

```sql
CREATE TABLE buildings (id TEXT PRIMARY KEY, name TEXT, body TEXT);
CREATE VIRTUAL TABLE buildings_fts USING fts5(name, body, content=buildings);
-- (same pattern for elements, foods, research)
```

---

## Spec A3: Wiki Tool Calling (Python — llm.py)

### Function Declaration

`GeminiAgent.__init__` registers a Gemini function declaration:

```python
search_wiki_fn = types.FunctionDeclaration(
    name="search_wiki",
    description="Search the ONI wiki for game data: building stats, element properties, food recipes, or research costs.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={"query": types.Schema(type=types.Type.STRING)},
        required=["query"],
    ),
)
```

### Decision Flow

`GeminiAgent.decide()` changes from a single `generate_content` call to a multi-turn loop:

1. Call `generate_content` with the state prompt + tool declaration.
2. If the response contains a `function_call` for `search_wiki`:
   - Execute the FTS query against `data/wiki.db`.
   - Return the top 3 matching rows (name + body, truncated to 300 chars each).
   - Send the result back as a `function_response` part.
   - Repeat (max 2 wiki calls per decision).
3. Once the response contains no function call, parse the text as the action JSON.

### Search Implementation

```python
def _search_wiki(self, query: str) -> str:
    conn = sqlite3.connect("data/wiki.db")
    results = []
    for table in ("buildings", "elements", "foods", "research"):
        rows = conn.execute(
            f"SELECT name, body FROM {table}_fts WHERE {table}_fts MATCH ? LIMIT 2",
            (query,)
        ).fetchall()
        for name, body in rows:
            results.append(f"{name}: {body[:300]}")
    conn.close()
    return "\n\n".join(results[:3]) if results else "No results found."
```

### Fallback

If `data/wiki.db` does not exist, `_search_wiki` returns `"Wiki database not available."` — the agent continues without it. No startup error.

### Cost Impact

Each wiki call adds one extra round-trip to Gemini. With `thinking_budget=0` and short function responses, this is ~100–300ms additional latency. Max 2 calls per tick bounds the worst case.

---

## Prompt Changes

`_format_state()` in `llm.py` gains a tiles summary section:

```
Tile window: x=105 y=183 w=62 h=32 (3844 tiles)
Sample elements near base: Sandstone, Vacuum, Oxygen, Dirt
```

The full `tiles.data` array is NOT included in the text prompt — it would be too large. Instead, a compact summary is generated:
- Count of each unique element in the window
- Top 5 elements by count

This gives Gemini spatial awareness without token explosion.

---

## Protocol Change

The state JSON gains one new top-level field `tiles`. No changes to the action protocol. No new action types.

---

## Files Changed

| File | Change |
|------|--------|
| `mod/ONIBridge/src/StateSerializer.cs` | Add `GetTiles()`, add `tiles` to `Serialize()` |
| `src/agent/llm.py` | Add tool declaration, multi-turn wiki loop, tiles summary in prompt |
| `src/agent/protocol.py` | No change |
| `scripts/build_wiki_db.py` | New file — one-time wiki scraper |
| `data/wiki.db` | New file — gitignored |
| `.gitignore` | Add `data/wiki.db` |

---

## Out of Scope

- Full tile array in the Gemini prompt (token cost too high)
- Temperature per tile (Phase 2 expansion)
- Automatic wiki DB refresh (manual re-run of scraper is sufficient)
- Real-time wiki HTTP calls (SQLite is faster and offline-safe)
