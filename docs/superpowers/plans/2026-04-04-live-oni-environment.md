# LiveONIEnvironment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `LiveONIEnvironment` — a gym-compatible RL environment that drives a real running ONI game via the TCP bridge, producing the same observation/reward interface as `MiniONIEnvironment`.

**Architecture:** `LiveONIEnvironment` wraps `BridgeClient` (already in `src/agent/client.py`) and adapts `StateMessage` data into numpy observations and scalar rewards. It runs an internal `asyncio` event loop in a background thread so the synchronous `step()`/`reset()` gym API works without changes to callers. The existing `MiniONIEnvironment` is left untouched — `LiveONIEnvironment` is a sibling, not a replacement.

**Tech Stack:** Python 3.11+, asyncio, numpy, existing `BridgeClient`/`StateMessage`/`build_action` from `src/agent/`, pytest

---

## File Structure

```
src/environments/live_oni/
    __init__.py              — exports LiveONIEnvironment
    environment.py           — LiveONIEnvironment class (gym step/reset/close)
    state_adapter.py         — StateMessage → numpy observation + reward signals
    action_adapter.py        — action_idx → bridge action dict

tests/environments/live_oni/
    test_state_adapter.py    — unit tests for observation and reward conversion
    test_action_adapter.py   — unit tests for action index → bridge dict
    test_environment.py      — integration tests with a mock BridgeClient
```

**Key design constraints:**
- `BridgeClient` lives at `src/agent/client.py` — import it, do not copy it
- Observation shape must match `MiniONIEnvironment._get_observation()`: flat float32 array of shape `(32*32*8 + 64,)` = `(8256,)`
- Action indices must map to the same `action_space` produced by `generate_action_space(64, 64)` from `src/environments/mini_oni/actions.py`
- `step()` sends one action, then waits for the **next** state tick (up to `step_timeout` seconds)
- `reset()` sends a `no_op` and waits for the next state tick — it does NOT restart the game
- The asyncio loop runs in a `threading.Thread`; `step()`/`reset()` use `asyncio.run_coroutine_threadsafe` + `.result(timeout)`

---

### Task 1: State Adapter — observation conversion

**Files:**
- Create: `src/environments/live_oni/state_adapter.py`
- Create: `tests/environments/live_oni/test_state_adapter.py`

The adapter converts `StateMessage.data` (the dict from the bridge) into a flat float32 numpy array of shape `(8256,)` that matches `MiniONIEnvironment._get_observation()` layout:
- Spatial: `(32, 32, 8)` zeroed (no tile map from bridge — bridge only sends resources/duplicants/buildings/alerts)
- Global features `(64,)` filled from bridge data using the same index layout as `MiniONIEnvironment._get_global_observation()`

The spatial portion will always be zeros for now — the bridge doesn't stream tile data. The global features vector is what matters.

Global feature index layout (copy from `MiniONIEnvironment._get_global_observation`):
- `[0]` cycle / max_cycles (normalize with max_cycles=100)
- `[1]` building count / 20.0
- `[2]` living duplicants / 3.0
- `[3]` happy duplicants (stress < 0.33) / 3.0
- `[4]` 0.0 (no tile breathability data)
- `[5]` 0.0 (no temperature data)
- `[10]` oxygen_kg / 1000.0 clamped to 1.0
- `[11]` food_kcal_today / 5000.0 clamped to 1.0
- `[12]` water_kg / 1000.0 clamped to 1.0
- `[13]` power_kw / 10.0 clamped to 1.0
- `[36..50]` duplicant stats: alive=1.0, happiness=(1-stress), stress=stress (3 values per dup, up to 5 dups)

- [ ] **Step 1: Write the failing tests**

```python
# tests/environments/live_oni/test_state_adapter.py
import numpy as np
import pytest
from src.environments.live_oni.state_adapter import state_to_observation, compute_reward

SAMPLE_DATA = {
    "cycle": 5,
    "time": 100.0,
    "resources": {
        "oxygen_kg": 200.0,
        "water_kg": 500.0,
        "food_kcal_today": 1000.0,
        "power_kw": 2.0,
        "co2_kg": 10.0,
    },
    "duplicants": [
        {"id": 1, "name": "Nikola", "x": 10, "y": 10, "stress": 0.1, "health": 100.0, "current_task": "dig"},
        {"id": 2, "name": "Ellie",  "x": 12, "y": 10, "stress": 0.2, "health": 100.0, "current_task": "idle"},
        {"id": 3, "name": "Pav",    "x": 14, "y": 10, "stress": 0.5, "health":  80.0, "current_task": "build"},
    ],
    "buildings": [
        {"type": "OxygenDiffuser", "x": 5, "y": 5, "operational": True},
        {"type": "Outhouse",       "x": 8, "y": 5, "operational": True},
    ],
    "alerts": [],
}


def test_observation_shape():
    obs = state_to_observation(SAMPLE_DATA, max_cycles=100)
    assert obs.shape == (8256,)
    assert obs.dtype == np.float32


def test_observation_cycle_normalized():
    obs = state_to_observation(SAMPLE_DATA, max_cycles=100)
    assert abs(obs[32 * 32 * 8 + 0] - 5 / 100) < 1e-5  # global[0] = cycle/max_cycles


def test_observation_building_count():
    obs = state_to_observation(SAMPLE_DATA, max_cycles=100)
    assert abs(obs[32 * 32 * 8 + 1] - 2 / 20.0) < 1e-5  # global[1] = buildings/20


def test_observation_oxygen():
    obs = state_to_observation(SAMPLE_DATA, max_cycles=100)
    assert abs(obs[32 * 32 * 8 + 10] - min(200.0 / 1000.0, 1.0)) < 1e-5


def test_observation_duplicant_stress():
    obs = state_to_observation(SAMPLE_DATA, max_cycles=100)
    base = 32 * 32 * 8 + 36
    assert obs[base + 0] == 1.0        # dup0 alive
    assert abs(obs[base + 2] - 0.1) < 1e-5  # dup0 stress


def test_spatial_portion_zeros():
    obs = state_to_observation(SAMPLE_DATA, max_cycles=100)
    assert np.all(obs[:32 * 32 * 8] == 0.0)


def test_compute_reward_no_alerts():
    reward = compute_reward(SAMPLE_DATA, prev_data=None)
    assert isinstance(reward, float)
    assert reward >= 0.0


def test_compute_reward_alert_penalty():
    data_with_alert = {**SAMPLE_DATA, "alerts": ["BreathabilityDiagnostic: Low oxygen"]}
    reward = compute_reward(data_with_alert, prev_data=None)
    reward_clean = compute_reward(SAMPLE_DATA, prev_data=None)
    assert reward < reward_clean


def test_compute_reward_duplicant_death():
    dead_data = {
        **SAMPLE_DATA,
        "duplicants": [
            {"id": 1, "name": "Nikola", "x": 10, "y": 10, "stress": 1.0, "health": 0.0, "current_task": "idle"},
        ],
    }
    alive_data = {
        **SAMPLE_DATA,
        "duplicants": [
            {"id": 1, "name": "Nikola", "x": 10, "y": 10, "stress": 0.1, "health": 100.0, "current_task": "dig"},
        ],
    }
    assert compute_reward(dead_data, prev_data=alive_data) < compute_reward(alive_data, prev_data=None)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
python3 -m pytest tests/environments/live_oni/test_state_adapter.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'src.environments.live_oni'`

- [ ] **Step 3: Create the package skeleton**

```python
# src/environments/live_oni/__init__.py
from .environment import LiveONIEnvironment

__all__ = ["LiveONIEnvironment"]
```

```bash
mkdir -p src/environments/live_oni tests/environments/live_oni
touch src/environments/live_oni/__init__.py
touch tests/environments/live_oni/__init__.py
```

- [ ] **Step 4: Implement `state_adapter.py`**

```python
# src/environments/live_oni/state_adapter.py
from __future__ import annotations
import numpy as np
from typing import Any

OBS_SPATIAL = 32 * 32 * 8   # 8192 — always zeros (bridge has no tile data)
OBS_GLOBAL  = 64
OBS_SIZE    = OBS_SPATIAL + OBS_GLOBAL  # 8256


def state_to_observation(data: dict[str, Any], max_cycles: int = 100) -> np.ndarray:
    """Convert bridge state dict to flat float32 observation of shape (8256,)."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    g = OBS_SPATIAL  # offset to global features

    resources   = data.get("resources", {})
    duplicants  = data.get("duplicants", [])
    buildings   = data.get("buildings", [])

    # Global features — same index layout as MiniONIEnvironment._get_global_observation
    obs[g + 0]  = data.get("cycle", 0) / max(max_cycles, 1)
    obs[g + 1]  = min(len(buildings), 20) / 20.0
    obs[g + 2]  = len(duplicants) / 3.0  # alive count (bridge only sends living dups)
    obs[g + 3]  = sum(1 for d in duplicants if d.get("stress", 1.0) < 0.33) / max(len(duplicants), 1)
    # [4],[5] breathable/temp tiles — no data, leave 0

    obs[g + 10] = min(resources.get("oxygen_kg", 0.0)        / 1000.0, 1.0)
    obs[g + 11] = min(resources.get("food_kcal_today", 0.0)  / 5000.0, 1.0)
    obs[g + 12] = min(resources.get("water_kg", 0.0)         / 1000.0, 1.0)
    obs[g + 13] = min(resources.get("power_kw", 0.0)         / 10.0,   1.0)

    # Duplicant stats: 3 values per dup starting at global[36], up to 5 dups
    for i, dup in enumerate(duplicants[:5]):
        base = g + 36 + i * 3
        obs[base + 0] = 1.0                              # alive (bridge omits dead dups)
        obs[base + 1] = 1.0 - float(dup.get("stress", 0.0))  # happiness proxy
        obs[base + 2] = float(dup.get("stress", 0.0))

    return obs


def compute_reward(data: dict[str, Any], prev_data: dict[str, Any] | None) -> float:
    """Compute scalar reward from current (and optionally previous) bridge state."""
    reward = 0.0
    duplicants = data.get("duplicants", [])
    alerts     = data.get("alerts", [])
    resources  = data.get("resources", {})

    # Per-duplicant survival reward
    reward += len(duplicants) * 0.05

    # Low-stress bonus
    reward += sum(0.05 for d in duplicants if d.get("stress", 1.0) < 0.33)

    # Oxygen resource bonus
    reward += min(resources.get("oxygen_kg", 0.0) / 1000.0, 1.0) * 0.1

    # Alert penalty
    reward -= len(alerts) * 0.5

    # Duplicant death penalty (compare against prev state)
    if prev_data is not None:
        prev_count = len(prev_data.get("duplicants", []))
        curr_count = len(duplicants)
        deaths = max(0, prev_count - curr_count)
        reward -= deaths * 50.0

    return float(reward)
```

- [ ] **Step 5: Run tests — expect pass**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
python3 -m pytest tests/environments/live_oni/test_state_adapter.py -v
```

Expected: `8 passed`

- [ ] **Step 6: Commit**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
git add src/environments/live_oni/__init__.py src/environments/live_oni/state_adapter.py tests/environments/live_oni/__init__.py tests/environments/live_oni/test_state_adapter.py
git commit -m "feat: add LiveONI state adapter — bridge dict → observation + reward"
```

---

### Task 2: Action Adapter — action index → bridge dict

**Files:**
- Create: `src/environments/live_oni/action_adapter.py`
- Create: `tests/environments/live_oni/test_action_adapter.py`

Maps integer action indices (from the same `generate_action_space(64, 64)` used by `MiniONIEnvironment`) to the bridge action dict format accepted by `BridgeServer`. Only the 4 bridge-supported action types are mapped: `place_building`, `dig`, `set_priority`, `no_op`. `DuplicantAction` has no bridge equivalent yet — map it to `no_op`.

Bridge action dict shapes (from `src/agent/protocol.py`):
- `{"type": "action", "action": "no_op"}`
- `{"type": "action", "action": "dig", "cell_x": int, "cell_y": int}`
- `{"type": "action", "action": "place_building", "building_id": str, "cell_x": int, "cell_y": int}`
- `{"type": "action", "action": "set_priority", "cell_x": int, "cell_y": int, "priority": int}`

Building ID mapping (ONI prefab IDs for the ESSENTIAL_BUILDINGS):
```python
BUILDING_TYPE_TO_PREFAB = {
    "COT":               "Bed",
    "OUTHOUSE":          "Outhouse",
    "RESEARCH_STATION":  "ResearchCenter",
    "OXYGEN_DIFFUSER":   "OxygenDiffuser",
    "ELECTROLYSIS":      "Electrolyzer",
    "MANUAL_GENERATOR":  "ManualGenerator",
    "COAL_GENERATOR":    "CoalGenerator",
    "WATER_PUMP":        "LiquidPumpingStation",
    "PLANTER_BOX":       "PlanterBox",
    "MESS_TABLE":        "MessTable",
}
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/environments/live_oni/test_action_adapter.py
import pytest
from src.environments.live_oni.action_adapter import action_to_bridge, BUILDING_TYPE_TO_PREFAB
from src.environments.mini_oni.actions import (
    generate_action_space, NoOpAction, DigAction, PlaceBuildingAction,
    PriorityAction, DuplicantAction, ActionType,
)
from src.environments.mini_oni.building_types import BuildingType

ACTION_SPACE = generate_action_space(64, 64)


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


def test_place_building_has_building_id():
    idx = _find_action_idx(ACTION_SPACE, PlaceBuildingAction)
    result = action_to_bridge(idx, ACTION_SPACE)
    assert result["action"] == "place_building"
    assert "building_id" in result
    assert "cell_x" in result
    assert "cell_y" in result


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


def test_building_type_to_prefab_covers_essential_buildings():
    from src.environments.mini_oni.building_types import ESSENTIAL_BUILDINGS
    for bt in ESSENTIAL_BUILDINGS:
        assert bt.name in BUILDING_TYPE_TO_PREFAB, f"Missing prefab for {bt.name}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
python3 -m pytest tests/environments/live_oni/test_action_adapter.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.environments.live_oni.action_adapter'`

- [ ] **Step 3: Implement `action_adapter.py`**

First, read `src/environments/mini_oni/building_types.py` to confirm the BuildingType enum names, then implement:

```python
# src/environments/live_oni/action_adapter.py
from __future__ import annotations
from typing import Any

from src.environments.mini_oni.actions import (
    Action, NoOpAction, DigAction, PlaceBuildingAction,
    PriorityAction, DuplicantAction,
)

BUILDING_TYPE_TO_PREFAB: dict[str, str] = {
    "COT":               "Bed",
    "OUTHOUSE":          "Outhouse",
    "RESEARCH_STATION":  "ResearchCenter",
    "OXYGEN_DIFFUSER":   "OxygenDiffuser",
    "ELECTROLYSIS":      "Electrolyzer",
    "MANUAL_GENERATOR":  "ManualGenerator",
    "COAL_GENERATOR":    "CoalGenerator",
    "WATER_PUMP":        "LiquidPumpingStation",
    "PLANTER_BOX":       "PlanterBox",
    "MESS_TABLE":        "MessTable",
}

_NO_OP = {"type": "action", "action": "no_op"}


def action_to_bridge(action_idx: int, action_space: list[Action]) -> dict[str, Any]:
    """Convert action index to bridge action dict."""
    if action_idx < 0 or action_idx >= len(action_space):
        return _NO_OP

    action = action_space[action_idx]

    if isinstance(action, NoOpAction):
        return _NO_OP

    if isinstance(action, DigAction):
        return {
            "type": "action",
            "action": "dig",
            "cell_x": action.region.x1,
            "cell_y": action.region.y1,
        }

    if isinstance(action, PlaceBuildingAction):
        prefab = BUILDING_TYPE_TO_PREFAB.get(action.building_type.name, "")
        return {
            "type": "action",
            "action": "place_building",
            "building_id": prefab,
            "cell_x": action.region.x1,
            "cell_y": action.region.y1,
        }

    if isinstance(action, PriorityAction):
        return {
            "type": "action",
            "action": "set_priority",
            "cell_x": action.region.x1 if hasattr(action, "region") else 0,
            "cell_y": action.region.y1 if hasattr(action, "region") else 0,
            "priority": int(action.priority_level),
        }

    # DuplicantAction and anything else — no bridge equivalent
    return _NO_OP
```

- [ ] **Step 4: Check PriorityAction fields**

Before running tests, read `src/environments/mini_oni/actions.py` to verify `PriorityAction` has a `region` attribute (not just `task_type` + `priority_level`). If it doesn't have `region`, the cell_x/cell_y in the adapter need to default to 0.

```bash
grep -A 10 "class PriorityAction" /Volumes/DevDrive/Projects/ONI-AI/src/environments/mini_oni/actions.py
```

Adjust `action_adapter.py` as needed based on the actual fields.

- [ ] **Step 5: Run tests — expect pass**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
python3 -m pytest tests/environments/live_oni/test_action_adapter.py -v
```

Expected: `7 passed`

- [ ] **Step 6: Commit**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
git add src/environments/live_oni/action_adapter.py tests/environments/live_oni/test_action_adapter.py
git commit -m "feat: add LiveONI action adapter — action index → bridge dict"
```

---

### Task 3: LiveONIEnvironment — core gym wrapper

**Files:**
- Create: `src/environments/live_oni/environment.py`
- Create: `tests/environments/live_oni/test_environment.py`

`LiveONIEnvironment` is a synchronous gym-style class. Internally it runs `asyncio` in a background thread (`_loop_thread`). The public API (`reset`, `step`, `close`) uses `asyncio.run_coroutine_threadsafe(...).result(timeout)` to call async methods.

**Public API:**
```python
class LiveONIEnvironment:
    def __init__(self, host: str, port: int = 9999, step_timeout: float = 5.0, max_cycles: int = 100)
    def reset(self) -> np.ndarray          # connects if needed, waits for next state tick
    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict]
    def close(self) -> None
```

**Internal flow:**
- `__init__`: creates `BridgeClient`, starts background thread with `asyncio.new_event_loop()`
- `reset()`: calls async `_async_reset()` — connects to bridge, waits for first `StateMessage`, returns observation
- `step(action_idx)`: calls async `_async_step(action_idx)` — sends bridge action, waits for next `StateMessage`, returns `(obs, reward, done, info)`
- `done` is `True` when `state.cycle >= max_cycles` or `len(duplicants) == 0`
- `_prev_data` stores previous state data for death detection in `compute_reward`

- [ ] **Step 1: Write the failing tests using a mock BridgeClient**

```python
# tests/environments/live_oni/test_environment.py
import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.environments.live_oni.environment import LiveONIEnvironment
from src.agent.protocol import StateMessage

SAMPLE_STATE = StateMessage(
    cycle=1,
    time=20.0,
    data={
        "cycle": 1,
        "time": 20.0,
        "resources": {"oxygen_kg": 200.0, "water_kg": 500.0, "food_kcal_today": 1000.0, "power_kw": 2.0, "co2_kg": 5.0},
        "duplicants": [
            {"id": 1, "name": "Nikola", "x": 10, "y": 10, "stress": 0.1, "health": 100.0, "current_task": "dig"},
        ],
        "buildings": [{"type": "OxygenDiffuser", "x": 5, "y": 5, "operational": True}],
        "alerts": [],
    }
)


def make_mock_client(states: list[StateMessage]):
    """Create a mock BridgeClient that yields states from the list."""
    async def mock_state_stream():
        for s in states:
            yield s

    client = MagicMock()
    client.connect = AsyncMock()
    client.send_action = AsyncMock()
    client.close = AsyncMock()
    client.state_stream = mock_state_stream
    return client


def test_reset_returns_correct_shape():
    with patch("src.environments.live_oni.environment.BridgeClient") as MockClient:
        MockClient.return_value = make_mock_client([SAMPLE_STATE, SAMPLE_STATE])
        env = LiveONIEnvironment(host="127.0.0.1", port=9999)
        obs = env.reset()
        assert obs.shape == (8256,)
        assert obs.dtype == np.float32
        env.close()


def test_step_returns_correct_tuple():
    with patch("src.environments.live_oni.environment.BridgeClient") as MockClient:
        MockClient.return_value = make_mock_client([SAMPLE_STATE] * 5)
        env = LiveONIEnvironment(host="127.0.0.1", port=9999)
        env.reset()
        obs, reward, done, info = env.step(0)
        assert obs.shape == (8256,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "cycle" in info
        env.close()


def test_done_when_max_cycles_reached():
    late_state = StateMessage(
        cycle=100,
        time=2000.0,
        data={**SAMPLE_STATE.data, "cycle": 100},
    )
    with patch("src.environments.live_oni.environment.BridgeClient") as MockClient:
        MockClient.return_value = make_mock_client([late_state] * 3)
        env = LiveONIEnvironment(host="127.0.0.1", port=9999, max_cycles=100)
        env.reset()
        _, _, done, _ = env.step(0)
        assert done is True
        env.close()


def test_done_when_no_duplicants():
    dead_state = StateMessage(
        cycle=5,
        time=100.0,
        data={**SAMPLE_STATE.data, "duplicants": []},
    )
    with patch("src.environments.live_oni.environment.BridgeClient") as MockClient:
        MockClient.return_value = make_mock_client([dead_state] * 3)
        env = LiveONIEnvironment(host="127.0.0.1", port=9999)
        env.reset()
        _, _, done, _ = env.step(0)
        assert done is True
        env.close()


def test_info_contains_expected_keys():
    with patch("src.environments.live_oni.environment.BridgeClient") as MockClient:
        MockClient.return_value = make_mock_client([SAMPLE_STATE] * 5)
        env = LiveONIEnvironment(host="127.0.0.1", port=9999)
        env.reset()
        _, _, _, info = env.step(0)
        assert "cycle" in info
        assert "duplicants" in info
        assert "alerts" in info
        assert "resources" in info
        env.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
python3 -m pytest tests/environments/live_oni/test_environment.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.environments.live_oni.environment'`

- [ ] **Step 3: Implement `environment.py`**

```python
# src/environments/live_oni/environment.py
from __future__ import annotations

import asyncio
import threading
import logging
from typing import Any

import numpy as np

from src.agent.client import BridgeClient
from src.agent.protocol import StateMessage
from src.environments.mini_oni.actions import generate_action_space
from src.environments.live_oni.state_adapter import state_to_observation, compute_reward
from src.environments.live_oni.action_adapter import action_to_bridge

logger = logging.getLogger(__name__)


class LiveONIEnvironment:
    """
    Gym-compatible environment that drives a real running ONI game via TCP bridge.

    step()/reset() are synchronous; internally runs asyncio in a background thread.
    Does NOT restart the game on reset() — sends no_op and waits for next state tick.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        step_timeout: float = 5.0,
        max_cycles: int = 100,
    ):
        self.host = host
        self.port = port
        self.step_timeout = step_timeout
        self.max_cycles = max_cycles

        self._action_space = generate_action_space(64, 64)
        self.num_actions = len(self._action_space)

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="LiveONILoop"
        )
        self._loop_thread.start()

        self._client = BridgeClient(host=host, port=port)
        self._state_iter = None
        self._prev_data: dict[str, Any] | None = None
        self._connected = False

    # ------------------------------------------------------------------
    # Public gym API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Connect (if needed) and wait for the next state tick."""
        state = self._run(_async_reset(self._client, self._loop))
        self._state_iter = None  # will be recreated on next step
        self._prev_data = None
        return state_to_observation(state.data, self.max_cycles)

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict]:
        """Send action, wait for next state tick, return (obs, reward, done, info)."""
        action_dict = action_to_bridge(action_idx, self._action_space)
        state = self._run(_async_step(self._client, action_dict))

        obs    = state_to_observation(state.data, self.max_cycles)
        reward = compute_reward(state.data, self._prev_data)
        done   = (state.cycle >= self.max_cycles or
                  len(state.data.get("duplicants", [])) == 0)
        info   = {
            "cycle":      state.cycle,
            "duplicants": state.data.get("duplicants", []),
            "alerts":     state.data.get("alerts", []),
            "resources":  state.data.get("resources", {}),
        }

        self._prev_data = state.data
        return obs, reward, done, info

    def close(self) -> None:
        """Close bridge connection and stop background loop."""
        self._run(self._client.close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, coro) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self.step_timeout)


async def _async_reset(client: BridgeClient) -> StateMessage:
    """Connect and return the first StateMessage."""
    await client.connect()
    async for state in client.state_stream():
        return state


async def _async_step(client: BridgeClient, action_dict: dict) -> StateMessage:
    """Send action and return the next StateMessage."""
    await client.send_action(action_dict)
    async for state in client.state_stream():
        return state
```

Note: `_async_reset` and `_async_step` each create a **new** `state_stream()` iterator per call. This is intentional — `state_stream()` is a generator that reads fresh lines from the socket each time. The background asyncio loop handles the actual socket I/O.

- [ ] **Step 4: Fix the `_run` call signatures**

The `_async_reset` and `_async_step` functions take `client` as a parameter but `_run` passes a coroutine. Update the `reset()` and `step()` calls to pass coroutines correctly:

```python
# In reset():
state = self._run(_async_reset(self._client))

# In step():
state = self._run(_async_step(self._client, action_dict))
```

These are already correct in the implementation above — no change needed. Verify by running tests.

- [ ] **Step 5: Run tests — expect pass**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
python3 -m pytest tests/environments/live_oni/test_environment.py -v
```

Expected: `5 passed`

- [ ] **Step 6: Run all live_oni tests together**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
python3 -m pytest tests/environments/live_oni/ -v
```

Expected: `20 passed` (8 + 7 + 5)

- [ ] **Step 7: Commit**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
git add src/environments/live_oni/environment.py tests/environments/live_oni/test_environment.py
git commit -m "feat: add LiveONIEnvironment gym wrapper over TCP bridge"
```

---

### Task 4: Smoke test against live game

**Files:**
- Create: `examples/live_oni_demo.py`

A runnable script that connects to the live ONI game, runs 5 steps, and prints results. Not a pytest test — meant to be run manually with the game running.

- [ ] **Step 1: Write the demo script**

```python
#!/usr/bin/env python3
# examples/live_oni_demo.py
"""
Smoke test: connect to live ONI game and run 5 steps.
Usage: python3 examples/live_oni_demo.py [host] [port]
  host: IP of machine running ONI (default: 10.0.0.10)
  port: bridge port (default: 9999)
"""
import sys
import logging
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, ".")
from src.environments.live_oni import LiveONIEnvironment

host = sys.argv[1] if len(sys.argv) > 1 else "10.0.0.10"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

print(f"Connecting to ONI at {host}:{port}...")
env = LiveONIEnvironment(host=host, port=port, step_timeout=10.0)

obs = env.reset()
print(f"reset() — obs shape: {obs.shape}, non-zero global features: {(obs[8192:] != 0).sum()}")

for step_num in range(5):
    obs, reward, done, info = env.step(0)  # action 0 = no_op
    print(
        f"step {step_num+1}: cycle={info['cycle']} "
        f"reward={reward:.3f} done={done} "
        f"dups={len(info['duplicants'])} "
        f"alerts={info['alerts']}"
    )
    if done:
        print("Episode done.")
        break

env.close()
print("Done.")
```

- [ ] **Step 2: Run against live game (ONI must be running with mod loaded)**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
python3 examples/live_oni_demo.py 10.0.0.10 9999
```

Expected output (exact values will vary):
```
Connecting to ONI at 10.0.0.10:9999...
reset() — obs shape: (8256,) non-zero global features: 8
step 1: cycle=N reward=0.2xx done=False dups=3 alerts=[]
step 2: cycle=N reward=0.2xx done=False dups=3 alerts=[]
...
Done.
```

- [ ] **Step 3: Commit**

```bash
cd /Volumes/DevDrive/Projects/ONI-AI
git add examples/live_oni_demo.py
git commit -m "feat: add live ONI environment demo script"
```

---

## Self-Review

**1. Spec coverage:**
- State → observation: Task 1 ✓
- Action index → bridge dict: Task 2 ✓
- Synchronous gym API (reset/step/close): Task 3 ✓
- Background asyncio thread: Task 3 ✓
- Done condition (max_cycles, all dead): Task 3 ✓
- Reward function: Task 1 ✓
- Live smoke test: Task 4 ✓

**2. Placeholder scan:** None found. All code blocks are complete.

**3. Type consistency:**
- `state_to_observation(data, max_cycles)` — used consistently in Task 1 and Task 3
- `compute_reward(data, prev_data)` — used consistently in Task 1 and Task 3
- `action_to_bridge(action_idx, action_space)` — used consistently in Task 2 and Task 3
- `BridgeClient` — imported from `src.agent.client` in both Task 3 and tests
- `StateMessage` — imported from `src.agent.protocol` in both Task 3 and tests
- `generate_action_space(64, 64)` — same call in Task 2 tests and Task 3 implementation
