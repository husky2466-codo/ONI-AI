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
