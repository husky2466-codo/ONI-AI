import json
import os
import pytest
from fastapi.testclient import TestClient

# Point at a temp profiles file before importing the app
os.environ["LLM_PROFILES_PATH"] = "/tmp/test_profiles.json"

from examples.dashboard.server import app

client = TestClient(app)


def setup_function():
    # Write a clean profiles file before each test
    profiles = {
        "active_id": "default",
        "profiles": [
            {
                "id": "default",
                "name": "Default",
                "endpoint_url": "http://10.0.0.69:8000/v1",
                "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
                "api_key": "",
                "vision_enabled": False,
            }
        ],
    }
    with open("/tmp/test_profiles.json", "w") as f:
        json.dump(profiles, f)


def teardown_function():
    if os.path.exists("/tmp/test_profiles.json"):
        os.remove("/tmp/test_profiles.json")


def test_get_profiles():
    r = client.get("/config/profiles")
    assert r.status_code == 200
    data = r.json()
    assert "profiles" in data
    assert "active_id" in data
    assert len(data["profiles"]) == 1


def test_add_profile():
    r = client.post("/config/profiles", json={
        "name": "DGX-B",
        "endpoint_url": "http://192.168.3.20:8000/v1",
        "model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        "api_key": "",
        "vision_enabled": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "id" in data

    r2 = client.get("/config/profiles")
    assert len(r2.json()["profiles"]) == 2


def test_update_profile():
    r = client.get("/config/profiles")
    pid = r.json()["profiles"][0]["id"]

    r2 = client.put(f"/config/profiles/{pid}", json={
        "name": "Updated",
        "endpoint_url": "http://10.0.0.69:8000/v1",
        "model": "new-model",
        "api_key": "sk-test",
        "vision_enabled": True,
    })
    assert r2.status_code == 200
    assert r2.json()["ok"] is True

    r3 = client.get("/config/profiles")
    p = next(p for p in r3.json()["profiles"] if p["id"] == pid)
    assert p["model"] == "new-model"
    assert p["vision_enabled"] is True


def test_delete_non_active_profile():
    # Add a second profile first
    r = client.post("/config/profiles", json={
        "name": "Temp",
        "endpoint_url": "http://x/v1",
        "model": "m",
        "api_key": "",
        "vision_enabled": False,
    })
    pid = r.json()["id"]

    r2 = client.delete(f"/config/profiles/{pid}")
    assert r2.status_code == 200
    assert r2.json()["ok"] is True

    r3 = client.get("/config/profiles")
    ids = [p["id"] for p in r3.json()["profiles"]]
    assert pid not in ids


def test_delete_active_profile_rejected():
    r = client.get("/config/profiles")
    active_id = r.json()["active_id"]
    r2 = client.delete(f"/config/profiles/{active_id}")
    assert r2.status_code == 400


def test_set_active_profile():
    r = client.post("/config/profiles", json={
        "name": "DGX-B",
        "endpoint_url": "http://192.168.3.20:8000/v1",
        "model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        "api_key": "",
        "vision_enabled": False,
    })
    pid = r.json()["id"]

    r2 = client.post(f"/config/profiles/{pid}/activate")
    assert r2.status_code == 200
    assert r2.json()["ok"] is True

    r3 = client.get("/config/profiles")
    assert r3.json()["active_id"] == pid


def test_get_game_config():
    r = client.get("/config/game")
    assert r.status_code == 200
    data = r.json()
    assert "host" in data
    assert "port" in data


def test_set_game_config():
    r = client.post("/config/game", json={"host": "10.0.0.99", "port": 9998})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r2 = client.get("/config/game")
    assert r2.json()["host"] == "10.0.0.99"
    assert r2.json()["port"] == 9998
