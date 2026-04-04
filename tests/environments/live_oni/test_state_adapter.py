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
    assert abs(obs[32 * 32 * 8 + 0] - 5 / 100) < 1e-5


def test_observation_building_count():
    obs = state_to_observation(SAMPLE_DATA, max_cycles=100)
    assert abs(obs[32 * 32 * 8 + 1] - 2 / 20.0) < 1e-5


def test_observation_oxygen():
    obs = state_to_observation(SAMPLE_DATA, max_cycles=100)
    assert abs(obs[32 * 32 * 8 + 10] - min(200.0 / 1000.0, 1.0)) < 1e-5


def test_observation_duplicant_stress():
    obs = state_to_observation(SAMPLE_DATA, max_cycles=100)
    base = 32 * 32 * 8 + 36
    assert obs[base + 0] == 1.0
    assert abs(obs[base + 2] - 0.1) < 1e-5


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


def test_observation_empty_duplicants():
    data = {**SAMPLE_DATA, "duplicants": []}
    obs = state_to_observation(data, max_cycles=100)
    assert obs.shape == (8256,)
    assert obs[32 * 32 * 8 + 2] == 0.0   # dup count
    assert obs[32 * 32 * 8 + 3] == 0.0   # happy dups


def test_observation_empty_buildings():
    data = {**SAMPLE_DATA, "buildings": []}
    obs = state_to_observation(data, max_cycles=100)
    assert obs[32 * 32 * 8 + 1] == 0.0   # building count


def test_compute_reward_empty_duplicants():
    data = {**SAMPLE_DATA, "duplicants": []}
    reward = compute_reward(data, prev_data=None)
    assert isinstance(reward, float)
    # no dups → no survival or stress reward; oxygen bonus still applies
    assert abs(reward - 0.02) < 1e-5
