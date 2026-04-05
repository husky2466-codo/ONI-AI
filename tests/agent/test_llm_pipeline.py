# tests/agent/test_llm_pipeline.py
from unittest.mock import MagicMock, patch
import pytest
from src.agent.llm import LLMAgent


def _make_agent():
    return LLMAgent(endpoint_url="http://localhost:9999/v1", model="test", api_key="x")


def _make_state():
    return {
        "cycle": 1, "time": 10.0,
        "resources": {"oxygen_kg": 5.0, "water_kg": 20.0, "food_kcal": 3000,
                      "power_kw": 0.0, "co2_kg": 0.0},
        "duplicants": [], "buildings": [], "alerts": [],
    }


def test_decide_returns_tuple():
    agent = _make_agent()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"type": "action", "action": "no_op"}'
    mock_response.usage = None
    with patch.object(agent._client.chat.completions, "create", return_value=mock_response):
        result = agent.decide(_make_state())
    assert isinstance(result, tuple), "decide() must return a tuple"
    assert len(result) == 3
    action, prompt, raw = result
    assert isinstance(action, dict)
    assert isinstance(prompt, str)
    assert isinstance(raw, str)


def test_decide_prompt_contains_cycle():
    agent = _make_agent()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"type": "action", "action": "no_op"}'
    mock_response.usage = None
    with patch.object(agent._client.chat.completions, "create", return_value=mock_response):
        _, prompt, _ = agent.decide(_make_state())
    assert "Cycle: 1" in prompt


def test_decide_raw_response_captured():
    agent = _make_agent()
    raw_text = '{"type": "action", "action": "no_op"}'
    mock_response = MagicMock()
    mock_response.choices[0].message.content = raw_text
    mock_response.usage = None
    with patch.object(agent._client.chat.completions, "create", return_value=mock_response):
        _, _, raw = agent.decide(_make_state())
    assert raw == raw_text


def test_decide_failure_returns_tuple_with_noop():
    agent = _make_agent()
    with patch.object(agent._client.chat.completions, "create", side_effect=Exception("timeout")):
        result = agent.decide(_make_state())
    assert isinstance(result, tuple)
    action, prompt, raw = result
    assert action["action"] == "no_op"
    assert isinstance(prompt, str)
    assert raw == ""
