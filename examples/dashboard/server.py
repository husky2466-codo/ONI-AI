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
from contextlib import asynccontextmanager

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

# Read API key from env (set once when you start the dashboard)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


# ---------------------------------------------------------------------------
# Runner process manager
# ---------------------------------------------------------------------------

_runner_proc: subprocess.Popen | None = None


def runner_running() -> bool:
    return _runner_proc is not None and _runner_proc.poll() is None


def start_runner() -> dict:
    global _runner_proc
    if runner_running():
        return {"ok": False, "error": "Runner already running", "pid": _runner_proc.pid}
    if not GOOGLE_API_KEY:
        return {"ok": False, "error": "GOOGLE_API_KEY not set in dashboard environment"}

    cmd = [
        sys.executable, "-m", "src.agent.runner",
        "--host", GAME_HOST,
        "--port", str(GAME_PORT),
        "--api-key", GOOGLE_API_KEY,
        "--log-episode", "episodes/run1.json",
    ]
    _runner_proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    logger.info("Runner started (pid %d)", _runner_proc.pid)
    return {"ok": True, "pid": _runner_proc.pid}


def stop_runner() -> dict:
    global _runner_proc
    if not runner_running():
        return {"ok": False, "error": "Runner not running"}
    pid = _runner_proc.pid
    _runner_proc.terminate()
    try:
        _runner_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _runner_proc.kill()
    _runner_proc = None
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


@app.get("/runner/status")
async def runner_status():
    return {"running": runner_running(), "pid": _runner_proc.pid if runner_running() else None}


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8181)
