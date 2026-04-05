# Building Telemetry Prompt Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the LLM prompt so the agent correctly understands and reasons about game-spawned starting buildings (Telepad, RationBox, etc.) and stops issuing false survival warnings about missing food when a ration box is present.

**Architecture:** All changes are in `src/agent/llm.py` — the system prompt string and `_format_state()` function. No C# changes needed; the data pipeline already sends the correct buildings. Three targeted edits: add a starting-buildings table to `SYSTEM_PROMPT`, update the food-source check in `_format_state()`, and add a `[SPAWNED]` tag to known pre-built structures in the buildings section.

**Tech Stack:** Python 3.14, pytest

---

## File Map

| File | Change |
|------|--------|
| `src/agent/llm.py` | Add starting-buildings table to `SYSTEM_PROMPT`; fix food-source check; add `[SPAWNED]` tag |
| `tests/agent/test_llm_wiki.py` | Add new tests for the updated `_format_state()` behaviour |

---

## Task 1: Fix the food-source survival check and add spawned-building tag

**Files:**
- Modify: `src/agent/llm.py:181-244`
- Test: `tests/agent/test_llm_wiki.py`

### Context

`_format_state()` currently has:

```python
buildings = data.get("buildings", [])
...
lines.append(f"Buildings on map: {len(buildings)}")
for b in buildings[:20]:
    op = "OK" if b.get("operational") else "OFFLINE"
    lines.append(f"  {b.get('type','?')} @ ({b.get('x','?')},{b.get('y','?')}) [{op}]")
if len(buildings) > 20:
    lines.append(f"  ...and {len(buildings) - 20} more")

# Summarize what survival buildings are still missing
built_types = {b.get("type") for b in buildings}
...
if "MicrobeMusher" not in built_types:
    survival.append("MicrobeMusher (no food production!)")
```

- [ ] **Step 1: Write failing tests**

Add to `tests/agent/test_llm_wiki.py`:

```python
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
    assert "food" not in output.lower() or "missing" not in output.lower()


def test_format_state_spawned_buildings_tagged():
    """Telepad and RationBox get [SPAWNED] tag in buildings section."""
    state = _make_state(buildings=[
        _make_building("Telepad", x=130, y=200),
        _make_building("RationBox", x=128, y=200),
        _make_building("Bed", x=116, y=201),
    ])
    output = _format_state(state)
    assert "Telepad [SPAWNED]" in output
    assert "RationBox [SPAWNED]" in output
    assert "Bed [SPAWNED]" not in output
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agent/test_llm_wiki.py::test_format_state_ration_box_suppresses_urgent_food_warning tests/agent/test_llm_wiki.py::test_format_state_no_food_at_all_gives_urgent_warning tests/agent/test_llm_wiki.py::test_format_state_microbe_musher_suppresses_food_warning tests/agent/test_llm_wiki.py::test_format_state_spawned_buildings_tagged -v
```

Expected: 4 failures — current code doesn't have `[SPAWNED]` tags and food check logic doesn't match.

- [ ] **Step 3: Update `_format_state()` in `src/agent/llm.py`**

Find the buildings block at lines ~213-244. Replace it entirely with:

```python
    _SPAWNED_TYPES = {"Telepad", "RationBox", "IceCooledFan", "StorageLocker"}

    lines.append("")
    lines.append(f"Buildings on map: {len(buildings)}")
    for b in buildings[:20]:
        op = "OK" if b.get("operational") else "OFFLINE"
        tag = " [SPAWNED]" if b.get("type") in _SPAWNED_TYPES else ""
        lines.append(f"  {b.get('type','?')}{tag} @ ({b.get('x','?')},{b.get('y','?')}) [{op}]")
    if len(buildings) > 20:
        lines.append(f"  ...and {len(buildings) - 20} more")

    # Summarize what survival buildings are still missing
    built_types = {b.get("type") for b in buildings}
    dup_count = len(dups) or 3
    survival = []
    if "Outhouse" not in built_types:
        survival.append("Outhouse (no toilet!)")
    if "Bed" not in built_types:
        survival.append(f"Bed x{dup_count} (no sleep!)")
    if "OxygenDiffuser" not in built_types and "Electrolyzer" not in built_types:
        survival.append("OxygenDiffuser (no oxygen production!)")

    _FOOD_PRODUCERS = {"MicrobeMusher", "GasRangeComplete", "ElectricGrillComplete"}
    if not (built_types & _FOOD_PRODUCERS):
        if "RationBox" in built_types:
            survival.append("MicrobeMusher (ration box has starting food, but build a cooker soon)")
        else:
            survival.append("MicrobeMusher (no food source!)")

    if "ManualGenerator" not in built_types:
        survival.append("ManualGenerator (no power!)")
    if "Battery" not in built_types:
        survival.append("Battery (no power storage!)")
    if "ResearchCenter" not in built_types:
        survival.append("ResearchCenter (no research!)")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agent/test_llm_wiki.py::test_format_state_ration_box_suppresses_urgent_food_warning tests/agent/test_llm_wiki.py::test_format_state_no_food_at_all_gives_urgent_warning tests/agent/test_llm_wiki.py::test_format_state_microbe_musher_suppresses_food_warning tests/agent/test_llm_wiki.py::test_format_state_spawned_buildings_tagged -v
```

Expected: 4 PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/agent/ -v
```

Expected: all tests pass (43+ tests).

- [ ] **Step 6: Commit**

```bash
git add src/agent/llm.py tests/agent/test_llm_wiki.py
git commit -m "fix: tag spawned buildings in prompt, fix false food-source warning when RationBox present"
```

---

## Task 2: Add starting-buildings table to SYSTEM_PROMPT

**Files:**
- Modify: `src/agent/llm.py:51-56` (after the `### Mid-game oxygen` section, before `## Survival rules`)

### Context

The current system prompt goes from the mid-game oxygen table directly to `## Survival rules` at line 57. The starting-buildings table needs to be inserted between them so the LLM knows what `Telepad`, `RationBox`, etc. are before it reads the survival rules.

- [ ] **Step 1: Write failing test**

Add to `tests/agent/test_llm_wiki.py`:

```python
def test_system_prompt_contains_starting_buildings_table():
    """SYSTEM_PROMPT must document Telepad and RationBox."""
    from src.agent.llm import SYSTEM_PROMPT
    assert "Telepad" in SYSTEM_PROMPT
    assert "RationBox" in SYSTEM_PROMPT
    assert "Printing Pod" in SYSTEM_PROMPT
    assert "Ration Box" in SYSTEM_PROMPT
    assert "pre-spawned" in SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agent/test_llm_wiki.py::test_system_prompt_contains_starting_buildings_table -v
```

Expected: FAIL — `Telepad` not in `SYSTEM_PROMPT`.

- [ ] **Step 3: Insert starting-buildings table into `SYSTEM_PROMPT`**

In `src/agent/llm.py`, find this exact line in `SYSTEM_PROMPT`:

```
## Survival rules
```

Insert the following block immediately before it (keep the `\n` separation):

```
## Starting buildings (pre-spawned — always present on a new map)

These buildings exist at game start on the Terra asteroid. Do NOT try to place them — they are already built.
They appear in the buildings list tagged [SPAWNED].

| type (in state) | In-game name    | Function |
|-----------------|-----------------|----------|
| Telepad         | Printing Pod    | Delivers duplicants or care packages every 3 cycles. Check printing_pod.cycles_until_next in state. Do not try to build or place this. |
| RationBox       | Ration Box      | Stores up to 20 kg food. Starting food supply is here. Does NOT produce food — needs a MicrobeMusher for long-term. |
| IceCooledFan    | Ice-Cooled Fan  | Cools nearby area using ice. Stops when ice runs out. |
| StorageLocker   | Storage Locker  | General-purpose raw material storage. |

```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/agent/test_llm_wiki.py::test_system_prompt_contains_starting_buildings_table -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/agent/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agent/llm.py tests/agent/test_llm_wiki.py
git commit -m "feat: add starting-buildings reference table to system prompt (Telepad, RationBox, etc.)"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|------------------|------|
| System prompt: add starting-buildings table with Telepad, RationBox, IceCooledFan, StorageLocker | Task 2 |
| Survival checklist: recognize RationBox as existing food supply | Task 1 |
| Survival checklist: mild warning (not urgent) when RationBox present but no cooker | Task 1 |
| Buildings prompt section: `[SPAWNED]` tag for known spawned types | Task 1 |
| Test: RationBox suppresses urgent food warning | Task 1 |
| Test: No food at all → urgent warning | Task 1 |
| Test: MicrobeMusher present → no food warning | Task 1 |
| Test: Telepad/RationBox tagged, Bed not tagged | Task 1 |
| Test: SYSTEM_PROMPT contains Telepad/RationBox documentation | Task 2 |

All spec requirements covered. No placeholders. No type inconsistencies (both tasks touch only `_format_state()` and `SYSTEM_PROMPT` — no shared interfaces to drift).
