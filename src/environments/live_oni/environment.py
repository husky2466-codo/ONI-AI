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
        self._prev_data: dict[str, Any] | None = None

    def reset(self) -> np.ndarray:
        """Connect (if needed) and wait for the next state tick."""
        state = self._run(_async_reset(self._client))
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
        try:
            self._run(self._client.close())
        except Exception as exc:
            logger.debug("Exception during close (ignored): %s", exc)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=2.0)

    def _run(self, coro) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self.step_timeout)


async def _async_reset(client: BridgeClient) -> StateMessage:
    """Connect and return the first StateMessage.

    Raises:
        RuntimeError: If the bridge closes before sending any state.
    """
    await client.connect()
    async for state in client.state_stream():
        return state
    raise RuntimeError("Bridge closed before sending a state message")


async def _async_step(client: BridgeClient, action_dict: dict) -> StateMessage:
    """Send action and return the next StateMessage.

    Raises:
        RuntimeError: If the bridge closes before sending any state.
    """
    await client.send_action(action_dict)
    async for state in client.state_stream():
        return state
    raise RuntimeError("Bridge closed before sending a state message")
