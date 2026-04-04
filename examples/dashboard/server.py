#!/usr/bin/env python3
"""
ONI Bridge Dashboard — FastAPI + WebSocket server.
Connects to the ONI game bridge and streams state to the browser.
Also forwards action commands from the browser back to the game.

Usage: python3 examples/dashboard/server.py [oni_host] [oni_port]
"""
import sys
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ONI_HOST = sys.argv[1] if len(sys.argv) > 1 else "10.0.0.10"
ONI_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

# ---------------------------------------------------------------------------
# Bridge connection — single shared connection, reconnects on drop
# ---------------------------------------------------------------------------

class BridgeConnection:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._lock = asyncio.Lock()
        self.last_state: dict = {}
        self.log: list[str] = []  # recent ACKs / errors, capped at 50

    def _append_log(self, msg: str):
        self.log.append(msg)
        if len(self.log) > 50:
            self.log.pop(0)

    async def connect(self):
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=5.0
            )
            self._append_log(f"Connected to {self.host}:{self.port}")
            logger.info("Connected to ONI bridge at %s:%d", self.host, self.port)
        except Exception as e:
            self._append_log(f"Connection failed: {e}")
            logger.warning("Could not connect to ONI bridge: %s", e)

    async def send_action(self, action: dict) -> str:
        """Send an action and return a status string."""
        if self._writer is None or self._writer.is_closing():
            return "Not connected to game"
        async with self._lock:
            try:
                payload = json.dumps(action).encode() + b"\n"
                self._writer.write(payload)
                await self._writer.drain()
                return f"Sent: {action.get('action')} → waiting for ACK"
            except Exception as e:
                self._append_log(f"Send error: {e}")
                return f"Send error: {e}"

    async def read_loop(self, broadcast_fn):
        """Continuously read from bridge, update last_state, broadcast to browsers."""
        while True:
            if self._reader is None:
                await asyncio.sleep(2)
                await self.connect()
                continue
            try:
                line = await asyncio.wait_for(self._reader.readline(), timeout=10.0)
                if not line:
                    self._append_log("Bridge disconnected — reconnecting...")
                    self._reader = self._writer = None
                    await asyncio.sleep(2)
                    await self.connect()
                    continue

                msg = json.loads(line.strip())

                if msg.get("type") == "state":
                    self.last_state = msg.get("data", {})
                    await broadcast_fn({"type": "state", "data": self.last_state})

                elif msg.get("type") == "ack":
                    log_entry = (
                        f"ACK {msg.get('action')}: "
                        f"{'OK' if msg.get('success') else 'FAIL'}"
                        + (f" — {msg.get('error')}" if msg.get("error") else "")
                    )
                    self._append_log(log_entry)
                    await broadcast_fn({"type": "ack", "log": self.log[-20:]})

            except asyncio.TimeoutError:
                # No data for 10s — send a ping via no_op
                await self.send_action({"type": "action", "action": "no_op"})
            except json.JSONDecodeError:
                pass
            except Exception as e:
                self._append_log(f"Read error: {e}")
                self._reader = self._writer = None
                await asyncio.sleep(2)
                await self.connect()


# ---------------------------------------------------------------------------
# WebSocket manager — fan-out to all connected browsers
# ---------------------------------------------------------------------------

class WSManager:
    def __init__(self):
        self.clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.remove(ws)


bridge = BridgeConnection(ONI_HOST, ONI_PORT)
ws_manager = WSManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await bridge.connect()
    asyncio.create_task(bridge.read_loop(ws_manager.broadcast))
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("examples/dashboard/index.html") as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    # Send current state immediately on connect
    if bridge.last_state:
        await ws.send_json({"type": "state", "data": bridge.last_state})
    if bridge.log:
        await ws.send_json({"type": "ack", "log": bridge.log[-20:]})
    try:
        while True:
            msg = await ws.receive_json()
            logger.info("Browser sent: %s", msg)
            action_type = msg.get("action")
            if not action_type:
                continue

            if action_type == "no_op":
                action = {"type": "action", "action": "no_op"}
            elif action_type == "dig":
                action = {
                    "type": "action", "action": "dig",
                    "cell_x": int(msg.get("x", 0)),
                    "cell_y": int(msg.get("y", 0)),
                }
            elif action_type == "cancel_dig":
                action = {
                    "type": "action", "action": "cancel_dig",
                    "cell_x": int(msg.get("x", 0)),
                    "cell_y": int(msg.get("y", 0)),
                }
            elif action_type == "place_building":
                action = {
                    "type": "action", "action": "place_building",
                    "building_id": msg.get("building_id", ""),
                    "cell_x": int(msg.get("x", 0)),
                    "cell_y": int(msg.get("y", 0)),
                }
            elif action_type == "set_priority":
                action = {
                    "type": "action", "action": "set_priority",
                    "cell_x": int(msg.get("x", 0)),
                    "cell_y": int(msg.get("y", 0)),
                    "priority": int(msg.get("priority", 5)),
                }
            else:
                continue

            await bridge.send_action(action)
            # Don't log here — the ACK from the game will arrive in read_loop and log there

    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8181)
