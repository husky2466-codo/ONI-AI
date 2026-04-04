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
    obs[g + 3]  = sum(1 for d in duplicants if d.get("stress", 0.0) < 0.33) / max(len(duplicants), 1)
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
    reward += sum(0.05 for d in duplicants if d.get("stress", 0.0) < 0.33)

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
