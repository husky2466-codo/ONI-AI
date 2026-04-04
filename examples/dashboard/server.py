#!/usr/bin/env python3
"""
ONI Bridge Dashboard — FastAPI + WebSocket server.

Subscribes to the runner's WebSocket relay (localhost:8182/ws) and
fans out state/ack updates to connected browsers.  Also forwards
manual action commands from the browser back to the runner relay.

Exposes HTTP endpoints so the dashboard UI can start/stop the runner:
  POST /runner/start   — spawn src.agent.runner as a subprocess
  POST /runner/stop    — kill it
  GET  /runner/status  — {"running": bool, "pid": int|null}

Usage: python3 examples/dashboard/server.py
"""
import os
import sys
import signal
import asyncio
import json
import logging
import subprocess
import uuid
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RELAY_HOST = "localhost"
RELAY_PORT = 8182
RELAY_URL  = f"ws://{RELAY_HOST}:{RELAY_PORT}/ws"

GAME_HOST  = "10.0.0.10"
GAME_PORT  = 9999


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

_PROFILES_PATH = Path(os.environ.get(
    "LLM_PROFILES_PATH",
    os.path.join(os.path.dirname(__file__), "llm_profiles.json")
))

_GAME_CONFIG: dict = {"host": GAME_HOST, "port": GAME_PORT}
_runner_start_time: float | None = None


def _load_profiles() -> dict:
    if _PROFILES_PATH.exists():
        with open(_PROFILES_PATH) as f:
            return json.load(f)
    default = {
        "active_id": "dgx-a",
        "profiles": [
            {
                "id": "dgx-a",
                "name": "DGX-A (Qwen2.5-72B)",
                "endpoint_url": "http://10.0.0.69:8000/v1",
                "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
                "api_key": "",
                "vision_enabled": False,
            },
            {
                "id": "dgx-b",
                "name": "DGX-B (Qwen2.5-72B)",
                "endpoint_url": "http://192.168.3.20:8000/v1",
                "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
                "api_key": "",
                "vision_enabled": False,
            },
            {
                "id": "gemini",
                "name": "Gemini 2.5 Flash",
                "endpoint_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "model": "gemini-2.5-flash",
                "api_key": "",
                "vision_enabled": False,
            },
        ]
    }
    _save_profiles(default)
    return default


def _save_profiles(data: dict) -> None:
    with open(_PROFILES_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Runner process manager
# ---------------------------------------------------------------------------

_runner_proc: subprocess.Popen | None = None


def runner_running() -> bool:
    return _runner_proc is not None and _runner_proc.poll() is None


def _build_runner_cmd() -> list[str]:
    """Build the runner subprocess command from the active LLM profile."""
    data = _load_profiles()
    active_id = data.get("active_id")
    profile = next((p for p in data.get("profiles", []) if p["id"] == active_id), None)

    cmd = [
        sys.executable, "-m", "src.agent.runner",
        "--host", _GAME_CONFIG["host"],
        "--port", str(_GAME_CONFIG["port"]),
        "--log-episode", "episodes/run1.json",
        "--auto-reload",
    ]
    if profile:
        endpoint = profile.get("endpoint_url", "")
        model = profile.get("model", "")
        api_key = profile.get("api_key", "") or os.environ.get("GOOGLE_API_KEY", "")
        if endpoint:
            cmd += ["--endpoint", endpoint]
        if model:
            cmd += ["--model", model]
        if api_key:
            cmd += ["--api-key", api_key]
        if profile.get("vision_enabled"):
            cmd += ["--vision"]
    else:
        # No profile — fall back to env var so the Start button still works
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if api_key:
            cmd += ["--api-key", api_key]
    return cmd


def start_runner() -> dict:
    global _runner_proc, _runner_start_time
    if runner_running():
        return {"ok": False, "error": "Runner already running", "pid": _runner_proc.pid}

    cmd = _build_runner_cmd()
    _runner_proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _runner_start_time = time.time()
    logger.info("Runner started (pid %d)", _runner_proc.pid)
    return {"ok": True, "pid": _runner_proc.pid}


def stop_runner() -> dict:
    global _runner_proc, _runner_start_time
    if not runner_running():
        return {"ok": False, "error": "Runner not running"}
    pid = _runner_proc.pid
    _runner_proc.terminate()
    try:
        _runner_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _runner_proc.kill()
    _runner_proc = None
    _runner_start_time = None
    logger.info("Runner stopped (pid %d)", pid)
    return {"ok": True, "pid": pid}


# ---------------------------------------------------------------------------
# Broadcast state to browser WebSocket clients
# ---------------------------------------------------------------------------

last_state:    dict       = {}
log_entries:   list[str]  = []
browser_clients: list[WebSocket] = []
_relay_ws = None


async def broadcast_browsers(data: dict) -> None:
    dead = []
    for ws in browser_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        browser_clients.remove(ws)


async def push_runner_status() -> None:
    await broadcast_browsers({
        "type":    "runner_status",
        "running": runner_running(),
        "pid":     _runner_proc.pid if runner_running() else None,
    })


# ---------------------------------------------------------------------------
# Relay subscriber loop — reconnects automatically
# ---------------------------------------------------------------------------

async def relay_loop() -> None:
    global last_state, log_entries, _relay_ws
    while True:
        try:
            logger.info("Connecting to runner relay at %s ...", RELAY_URL)
            async with websockets.connect(RELAY_URL) as ws:
                _relay_ws = ws
                logger.info("Relay connected.")
                await push_runner_status()
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") == "state":
                        last_state = msg.get("data", {})
                    elif msg.get("type") == "ack":
                        log_entries = msg.get("log", log_entries)
                    await broadcast_browsers(msg)
        except Exception as e:
            _relay_ws = None
            logger.warning("Relay disconnected: %s — retrying in 3s", e)
            await push_runner_status()
            await asyncio.sleep(3)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(relay_loop())
    yield
    stop_runner()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("examples/dashboard/index.html") as f:
        return f.read()


@app.post("/runner/start")
async def runner_start():
    result = start_runner()
    await push_runner_status()
    return JSONResponse(result)


@app.post("/runner/stop")
async def runner_stop():
    result = stop_runner()
    await push_runner_status()
    return JSONResponse(result)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    browser_clients.append(ws)

    # Send current state immediately on connect
    if last_state:
        await ws.send_json({"type": "state", "data": last_state})
    if log_entries:
        await ws.send_json({"type": "ack", "log": log_entries[-20:]})
    await ws.send_json({
        "type":    "runner_status",
        "running": runner_running(),
        "pid":     _runner_proc.pid if runner_running() else None,
    })

    try:
        while True:
            msg = await ws.receive_json()
            logger.info("Browser action: %s", msg)
            if _relay_ws is not None:
                try:
                    await _relay_ws.send(json.dumps(msg))
                except Exception as e:
                    logger.warning("Could not forward to relay: %s", e)
            else:
                logger.warning("Relay not connected — cannot forward action")
    except WebSocketDisconnect:
        if ws in browser_clients:
            browser_clients.remove(ws)


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

@app.get("/config/profiles")
async def get_profiles():
    return _load_profiles()


@app.post("/config/profiles")
async def add_profile(profile: dict):
    data = _load_profiles()
    safe = {
        "name": profile.get("name", ""),
        "endpoint_url": profile.get("endpoint_url", ""),
        "model": profile.get("model", ""),
        "api_key": profile.get("api_key", ""),
        "vision_enabled": bool(profile.get("vision_enabled", False)),
    }
    safe["id"] = str(uuid.uuid4())[:8]
    data["profiles"].append(safe)
    _save_profiles(data)
    return {"ok": True, "id": safe["id"]}


@app.put("/config/profiles/{profile_id}")
async def update_profile(profile_id: str, updates: dict):
    data = _load_profiles()
    for p in data["profiles"]:
        if p["id"] == profile_id:
            p.update(updates)
            p["id"] = profile_id  # prevent id overwrite
            _save_profiles(data)
            return {"ok": True}
    return JSONResponse({"ok": False, "error": "not found"}, status_code=404)


@app.delete("/config/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    data = _load_profiles()
    if data["active_id"] == profile_id:
        return JSONResponse({"ok": False, "error": "cannot delete active profile"}, status_code=400)
    data["profiles"] = [p for p in data["profiles"] if p["id"] != profile_id]
    _save_profiles(data)
    return {"ok": True}


@app.post("/config/profiles/{profile_id}/activate")
async def activate_profile(profile_id: str):
    data = _load_profiles()
    if not any(p["id"] == profile_id for p in data["profiles"]):
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    data["active_id"] = profile_id
    _save_profiles(data)
    return {"ok": True}


@app.get("/config/game")
async def get_game_config():
    return _GAME_CONFIG


@app.post("/config/game")
async def set_game_config(cfg: dict):
    global _GAME_CONFIG
    _GAME_CONFIG["host"] = cfg.get("host", _GAME_CONFIG["host"])
    _GAME_CONFIG["port"] = int(cfg.get("port", _GAME_CONFIG["port"]))
    return {"ok": True}


@app.get("/runner/status")
async def runner_status():
    uptime = None
    if runner_running() and _runner_start_time is not None:
        uptime = int(time.time() - _runner_start_time)
    return {
        "running": runner_running(),
        "pid": _runner_proc.pid if runner_running() else None,
        "uptime_seconds": uptime,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8181)
