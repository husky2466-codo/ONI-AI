# Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-page ONI dashboard with a 6-tab operator console surfacing all Phase 1 data (dupe skills/traits/hunger/bladder, research, printing pod, perimeter, reward) with agnostic LLM backend profile management.

**Architecture:** The existing `server.py` (FastAPI + WebSocket relay) gains new HTTP endpoints for config management and runner control. `index.html` is replaced entirely with a tabbed single-page app — all tabs live in one HTML file with tab switching in vanilla JS. A new `llm_profiles.json` config file stores backend profiles; the server reads it and exposes CRUD endpoints.

**Tech Stack:** Python 3 / FastAPI / uvicorn (server), vanilla JS + CSS (frontend), JSON file (profile persistence). No new Python dependencies beyond what's already installed.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `examples/dashboard/index.html` | **Replace** | Entire frontend — all 6 tabs, status bar, health strip, toolbar |
| `examples/dashboard/server.py` | **Modify** | Add config endpoints, profile management, runner uptime tracking |
| `examples/dashboard/llm_profiles.json` | **Create** | Persisted LLM backend profiles (gitignored by default, add to .gitignore) |
| `tests/dashboard/test_server.py` | **Create** | Tests for new server endpoints |

---

## Task 1: Add llm_profiles.json and server config endpoints

**Files:**
- Create: `examples/dashboard/llm_profiles.json`
- Modify: `examples/dashboard/server.py`
- Create: `tests/dashboard/test_server.py`

- [ ] **Step 1: Write failing tests for profile endpoints**

Create `tests/dashboard/test_server.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
pytest tests/dashboard/test_server.py -v 2>&1 | head -40
```

Expected: import errors or 404s — all FAIL.

- [ ] **Step 3: Create default llm_profiles.json**

Create `examples/dashboard/llm_profiles.json`:

```json
{
  "active_id": "dgx-a",
  "profiles": [
    {
      "id": "dgx-a",
      "name": "DGX-A",
      "endpoint_url": "http://10.0.0.69:8000/v1",
      "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
      "api_key": "",
      "vision_enabled": false
    }
  ]
}
```

- [ ] **Step 4: Add `__init__.py` to tests/dashboard**

```bash
mkdir -p tests/dashboard
touch tests/dashboard/__init__.py
```

- [ ] **Step 5: Add profile management and config endpoints to server.py**

Add the following to `examples/dashboard/server.py` after the existing imports and before the runner process manager section:

```python
import uuid
import time
from pathlib import Path

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
        "active_id": "default",
        "profiles": [{
            "id": "default",
            "name": "Default",
            "endpoint_url": f"http://{GAME_HOST}:8000/v1",
            "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
            "api_key": "",
            "vision_enabled": False,
        }]
    }
    _save_profiles(default)
    return default


def _save_profiles(data: dict) -> None:
    with open(_PROFILES_PATH, "w") as f:
        json.dump(data, f, indent=2)
```

Also update `start_runner()` to record start time — replace the existing `start_runner` function body's final two lines:

```python
    _runner_proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _runner_start_time = time.time()
    logger.info("Runner started (pid %d)", _runner_proc.pid)
    return {"ok": True, "pid": _runner_proc.pid}
```

And update `stop_runner()` to clear start time — add after `_runner_proc = None`:

```python
    _runner_start_time = None
```

Then add these endpoints at the end of the FastAPI routes section (before `if __name__ == "__main__"`):

```python
# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

@app.get("/config/profiles")
async def get_profiles():
    return _load_profiles()


@app.post("/config/profiles")
async def add_profile(profile: dict):
    data = _load_profiles()
    profile["id"] = str(uuid.uuid4())[:8]
    data["profiles"].append(profile)
    _save_profiles(data)
    return {"ok": True, "id": profile["id"]}


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
```

Note: remove the existing `@app.get("/runner/status")` route — it's being replaced above.

- [ ] **Step 6: Run tests — expect most to pass**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
pytest tests/dashboard/test_server.py -v
```

Expected: all tests PASS. Fix any failures before continuing.

- [ ] **Step 7: Add llm_profiles.json to .gitignore**

```bash
echo "examples/dashboard/llm_profiles.json" >> .gitignore
```

- [ ] **Step 8: Commit**

```bash
git add examples/dashboard/server.py examples/dashboard/llm_profiles.json \
        tests/dashboard/__init__.py tests/dashboard/test_server.py .gitignore
git commit -m "feat: add LLM profile management and config endpoints to dashboard server"
```

---

## Task 2: Replace index.html — shell, status bar, tab bar, health strip, toolbar

**Files:**
- Replace: `examples/dashboard/index.html`

This task builds the structural skeleton only — tab bodies are empty placeholders. No tab content yet.

- [ ] **Step 1: Replace index.html with the shell**

Write `examples/dashboard/index.html` in full:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ONI Dashboard</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d1117; color: #c9d1d9;
  font-family: 'Courier New', monospace; font-size: 13px;
  height: 100vh; overflow: hidden;
  display: flex; flex-direction: column;
}

/* ── Status bar ── */
#statusbar {
  display: flex; align-items: center; gap: 8px;
  padding: 4px 10px; background: #161b22;
  border-bottom: 1px solid #30363d; flex-shrink: 0; font-size: 11px;
  color: #8b949e;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: #f85149; flex-shrink: 0; }
.dot.on { background: #3fb950; }
#statusbar .sep { color: #30363d; }
#statusbar .upd { margin-left: auto; color: #484f58; }

/* ── Tab bar ── */
#tabbar {
  display: flex; background: #161b22;
  border-bottom: 1px solid #30363d; flex-shrink: 0;
}
#tabbar button {
  padding: 6px 14px; font-size: 11px; text-transform: uppercase;
  letter-spacing: .5px; color: #8b949e; border: none; border-right: 1px solid #21262d;
  background: transparent; cursor: pointer; font-family: inherit;
  border-bottom: 2px solid transparent;
}
#tabbar button:hover { color: #c9d1d9; }
#tabbar button.active { color: #58a6ff; border-bottom: 2px solid #58a6ff; background: #0d1117; }

/* ── Tab bodies ── */
#tab-content { flex: 1; overflow: hidden; min-height: 0; }
.tab-body { display: none; height: 100%; overflow-y: auto; padding: 6px; }
.tab-body.active { display: block; }

/* ── Colony Health strip ── */
#health-strip {
  display: flex; align-items: center; gap: 18px;
  padding: 5px 10px; background: #161b22;
  border-top: 1px solid #30363d; flex-shrink: 0; font-size: 11px;
}
#health-strip .hs-label { color: #484f58; text-transform: uppercase; letter-spacing: .5px; font-size: 10px; }
#health-strip .hs-val { font-weight: bold; }
#health-strip .hs-val.ok { color: #3fb950; }
#health-strip .hs-val.neg { color: #f85149; }
.obligation { background: #2d1f1f; color: #f0883e; padding: 2px 7px; border-radius: 2px; font-size: 10px; }
#health-strip .phase-info { margin-left: auto; color: #484f58; font-size: 11px; }

/* ── Toolbar ── */
#toolbar {
  display: flex; align-items: center; padding: 5px 8px;
  background: #0d1117; border-top: 1px solid #30363d;
  gap: 4px; flex-shrink: 0;
}
.tb-btn {
  background: #161b22; border: 1px solid #30363d; border-radius: 3px;
  padding: 3px 8px; font-size: 10px; color: #8b949e; cursor: pointer;
  font-family: inherit;
}
.tb-btn:hover { background: #21262d; color: #c9d1d9; }
.speed-group { display: flex; border: 1px solid #30363d; border-radius: 3px; overflow: hidden; margin-left: auto; }
.spd {
  background: #21262d; color: #8b949e; padding: 3px 9px;
  font-size: 11px; border-right: 1px solid #30363d; cursor: pointer;
}
.spd:last-child { border-right: none; }
.spd.active { background: #1f3a2b; color: #3fb950; font-weight: bold; }

/* ── Shared card styles ── */
.card {
  background: #161b22; border: 1px solid #30363d;
  border-radius: 4px; padding: 8px 10px;
}
.card h3 {
  font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
  color: #58a6ff; margin-bottom: 6px; padding-bottom: 4px;
  border-bottom: 1px solid #21262d;
}
.kv { display: flex; justify-content: space-between; padding: 2px 0; }
.kv .k { color: #8b949e; font-size: 12px; }
.kv .v { font-weight: bold; font-size: 12px; color: #c9d1d9; }
.kv .v.ok { color: #3fb950; }
.kv .v.warn { color: #f0883e; }
.kv .v.bad { color: #f85149; }
.progress-track { height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; margin: 4px 0; }
.progress-fill { height: 100%; border-radius: 3px; background: #58a6ff; }
</style>
</head>
<body>

<!-- Status bar -->
<div id="statusbar">
  <span class="dot" id="dot-relay"></span>
  <span id="relay-label" style="color:#c9d1d9">Connecting...</span>
  <span class="sep">|</span>
  <span id="chain-label" class="chain">dashboard:8181 → relay:8182 → game:9999</span>
  <span class="sep">|</span>
  <span class="dot" id="dot-runner"></span>
  <span id="runner-label">Runner: stopped</span>
  <span class="upd" id="upd-label"></span>
</div>

<!-- Tab bar -->
<div id="tabbar">
  <button class="active" data-tab="colony" onclick="switchTab('colony')">🏠 Colony</button>
  <button data-tab="duplicants" onclick="switchTab('duplicants')">👥 Duplicants</button>
  <button data-tab="research" onclick="switchTab('research')">🔬 Research</button>
  <button data-tab="perimeter" onclick="switchTab('perimeter')">📐 Perimeter</button>
  <button data-tab="log" onclick="switchTab('log')">📋 Log</button>
  <button data-tab="config" onclick="switchTab('config')">⚙️ Config</button>
</div>

<!-- Tab content -->
<div id="tab-content">
  <div class="tab-body active" id="tab-colony"><!-- Task 3 --></div>
  <div class="tab-body" id="tab-duplicants"><!-- Task 4 --></div>
  <div class="tab-body" id="tab-research"><!-- Task 5 --></div>
  <div class="tab-body" id="tab-perimeter"><!-- Task 6 --></div>
  <div class="tab-body" id="tab-log"><!-- Task 7 --></div>
  <div class="tab-body" id="tab-config"><!-- Task 8 --></div>
</div>

<!-- Colony Health strip -->
<div id="health-strip">
  <span class="hs-label">Colony Health</span>
  <span>Tick: <span class="hs-val ok" id="hs-tick">—</span></span>
  <span>Episode: <span class="hs-val ok" id="hs-episode">—</span></span>
  <span>Survival: <span class="hs-val ok" id="hs-survival">—</span></span>
  <span id="hs-obligations"></span>
  <span class="phase-info" id="hs-phase"></span>
</div>

<!-- Toolbar -->
<div id="toolbar">
  <button class="tb-btn" onclick="sendAction('base')">🏗️ Base</button>
  <button class="tb-btn" onclick="sendAction('o2')">💨 O2</button>
  <button class="tb-btn" onclick="sendAction('power')">⚡ Power</button>
  <button class="tb-btn" onclick="sendAction('dig')">⛏️ Dig</button>
  <button class="tb-btn" onclick="sendAction('priority')">📌 Pri</button>
  <button class="tb-btn" onclick="sendAction('noop')">⏭️ No-op</button>
  <div style="flex:1"></div>
  <span id="speed-label" style="color:#3fb950;font-size:11px;margin-right:6px">● 1x</span>
  <div class="speed-group">
    <div class="spd" onclick="setSpeed(0)">⏸</div>
    <div class="spd active" id="spd-1" onclick="setSpeed(1)">1x</div>
    <div class="spd" id="spd-2" onclick="setSpeed(2)">2x</div>
    <div class="spd" id="spd-3" onclick="setSpeed(3)">4x</div>
  </div>
  <button class="tb-btn" style="margin-left:6px" onclick="sendUiAction('open_settings')">⚙️ Settings</button>
</div>

<script>
// ── WebSocket connection ──
var ws = null;
var lastStateTime = null;
var state = {};
var sessionStats = { ticks:0, actions:0, ackOk:0, ackFail:0, noops:0, errors:0 };
var logEntries = [];
var rewardData = { tick: null, episode: 0, survival: null, obligations: [] };

function connect() {
  ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onopen = function() {
    document.getElementById('dot-relay').classList.add('on');
    document.getElementById('relay-label').textContent = 'Connected';
    document.getElementById('relay-label').style.color = '#c9d1d9';
  };
  ws.onclose = function() {
    document.getElementById('dot-relay').classList.remove('on');
    document.getElementById('relay-label').textContent = 'Disconnected';
    document.getElementById('relay-label').style.color = '#f85149';
    setTimeout(connect, 3000);
  };
  ws.onerror = function() {
    addLogEntry('error', 'WebSocket error');
  };
  ws.onmessage = function(e) {
    var msg = JSON.parse(e.data);
    handleMessage(msg);
  };
}

function handleMessage(msg) {
  if (msg.type === 'state') {
    state = msg.data || {};
    lastStateTime = Date.now();
    sessionStats.ticks++;
    renderAll();
  } else if (msg.type === 'ack') {
    var ok = msg.success !== false;
    if (ok) { sessionStats.ackOk++; } else { sessionStats.ackFail++; }
    addLogEntry(ok ? 'ack-ok' : 'ack-fail', (msg.action || '') + ' → ' + (ok ? 'success' : (msg.error || 'failed')));
    renderLog();
  } else if (msg.type === 'runner_status') {
    var running = msg.running;
    document.getElementById('dot-runner').className = 'dot' + (running ? ' on' : '');
    document.getElementById('runner-label').textContent = 'Runner: ' + (running ? 'running' : 'stopped');
    renderConfig();
  }
}

// ── Tab switching ──
function switchTab(name) {
  document.querySelectorAll('.tab-body').forEach(function(el) { el.classList.remove('active'); });
  document.querySelectorAll('#tabbar button').forEach(function(el) { el.classList.remove('active'); });
  document.getElementById('tab-' + name).classList.add('active');
  document.querySelector('#tabbar button[data-tab="' + name + '"]').classList.add('active');
  if (name === 'config') loadConfig();
}

// ── Actions ──
function sendAction(type) {
  if (!ws || ws.readyState !== 1) return;
  var action = { type: 'action', action: type };
  ws.send(JSON.stringify(action));
  sessionStats.actions++;
  addLogEntry('action', type);
  renderLog();
}

function sendUiAction(action) {
  if (!ws || ws.readyState !== 1) return;
  ws.send(JSON.stringify({ type: 'ui_action', action: action }));
}

function setSpeed(speed) {
  sendAction('set_speed');
  [0,1,2,3].forEach(function(s) {
    var el = document.getElementById('spd-' + s);
    if (el) el.classList.toggle('active', s === speed);
  });
}

// ── Render helpers ──
function setText(id, val) {
  var el = document.getElementById(id);
  if (el) el.textContent = val != null ? val : '—';
}

function renderAll() {
  renderColony();
  renderDuplicants();
  renderResearch();
  renderHealthStrip();
  updateTimestamp();
}

function updateTimestamp() {
  if (lastStateTime) {
    var sec = ((Date.now() - lastStateTime) / 1000).toFixed(1);
    setText('upd-label', 'updated ' + sec + 's ago');
  }
}
setInterval(updateTimestamp, 500);

// Stub render functions — filled in per task
function renderColony() {}
function renderDuplicants() {}
function renderResearch() {}
function renderPerimeter() {}
function renderLog() {}
function renderConfig() {}
function renderHealthStrip() {
  var r = rewardData;
  if (r.tick != null) {
    var tickEl = document.getElementById('hs-tick');
    tickEl.textContent = (r.tick >= 0 ? '+' : '') + r.tick.toFixed(2);
    tickEl.className = 'hs-val ' + (r.tick >= 0 ? 'ok' : 'neg');
  }
  setText('hs-episode', r.episode != null ? (r.episode >= 0 ? '+' : '') + r.episode.toFixed(1) : '—');
  if (r.survival != null) setText('hs-survival', r.survival);
  var obl = document.getElementById('hs-obligations');
  obl.textContent = '';
  (r.obligations || []).forEach(function(o) {
    var span = document.createElement('span');
    span.className = 'obligation';
    span.textContent = '⚠ ' + o;
    obl.appendChild(span);
  });
  setText('hs-phase', state.phase != null ? 'Phase ' + state.phase + ' · cycle ' + (state.cycle || 0) + '/100' : '');
}

// ── Log helper ──
function addLogEntry(type, msg) {
  logEntries.push({ ts: new Date().toLocaleTimeString('en',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'}), type: type, msg: msg });
  if (logEntries.length > 500) logEntries.shift();
}

connect();
</script>
</body>
</html>
```

- [ ] **Step 2: Open dashboard in browser and verify tab switching works**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
python3 examples/dashboard/server.py &
open http://localhost:8181
```

Expected: dark page with status bar, 6 tabs, empty bodies, health strip, toolbar. Tab clicks switch active tab. No JS errors in console.

Kill the server after verifying: `kill %1`

- [ ] **Step 3: Commit**

```bash
git add examples/dashboard/index.html
git commit -m "feat: dashboard shell — status bar, 6 tabs, health strip, toolbar"
```

---

## Task 3: Colony tab

**Files:**
- Modify: `examples/dashboard/index.html` — fill `#tab-colony` and `renderColony()`

- [ ] **Step 1: Add Colony tab HTML inside `#tab-colony`**

Replace `<!-- Task 3 -->` inside `<div class="tab-body active" id="tab-colony">` with:

```html
<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;height:100%">
  <!-- Left column -->
  <div style="display:flex;flex-direction:column;gap:6px">
    <div class="card">
      <h3>Game</h3>
      <div style="font-size:32px;color:#58a6ff;font-weight:bold;line-height:1" id="col-cycle">—</div>
      <div style="font-size:9px;color:#484f58;margin-bottom:6px">CYCLE</div>
      <div class="kv"><span class="k">Time</span><span class="v" id="col-time">—</span></div>
      <div class="kv"><span class="k">Phase</span><span class="v" id="col-phase">—</span></div>
      <div class="kv"><span class="k">Speed</span><span class="v" id="col-speed">—</span></div>
    </div>
    <div class="card">
      <h3>Resources</h3>
      <div id="col-resources"></div>
    </div>
  </div>
  <!-- Right column -->
  <div style="display:flex;flex-direction:column;gap:6px">
    <div class="card">
      <h3>Alerts</h3>
      <div id="col-alerts" style="font-size:12px;color:#484f58">No alerts</div>
    </div>
    <div class="card" style="flex:1">
      <h3>Tile Window</h3>
      <div id="col-tile-pos" style="font-size:10px;color:#484f58;margin-bottom:4px"></div>
      <canvas id="col-tile-canvas" style="image-rendering:pixelated;width:100%;max-width:256px"></canvas>
    </div>
  </div>
</div>
```

- [ ] **Step 2: Implement `renderColony()` in the script block**

Replace the stub `function renderColony() {}` with:

```javascript
var TILE_COLORS = {
  'Vacuum': '#0d1117',
  'Sandstone': '#c8a96e',
  'Dirt': '#8b6914',
  'Granite': '#7a7a8a',
  'Algae': '#3fb950',
  'Water': '#1f6feb',
  'Oxygen': '#58a6ff22',
  'CarbonDioxide': '#f0883e33',
  'Polluted Water': '#5a4a1a',
  'Polluted Oxygen': '#8b8b1a',
};

function renderColony() {
  if (!state) return;
  setText('col-cycle', state.cycle != null ? state.cycle : '—');
  setText('col-time', state.time != null ? state.time.toFixed(1) + 's' : '—');
  setText('col-phase', state.phase != null ? state.phase : '—');
  setText('col-speed', state.speed != null ? state.speed + 'x' : '—');

  // Resources
  var res = state.resources || {};
  var resEl = document.getElementById('col-resources');
  resEl.textContent = '';
  var rows = [
    ['O2', res.oxygen_kg, 'kg'],
    ['Water', res.water_kg, 'kg'],
    ['Food', res.food_kcal, 'kcal'],
    ['Power', res.power_kw, 'kW'],
    ['CO2', res.co2_kg, 'kg'],
  ];
  rows.forEach(function(r) {
    var div = document.createElement('div');
    div.className = 'kv';
    var k = document.createElement('span'); k.className = 'k'; k.textContent = r[0];
    var v = document.createElement('span'); v.className = 'v'; v.textContent = r[1] != null ? r[1].toFixed(1) + ' ' + r[2] : '—';
    div.appendChild(k); div.appendChild(v);
    resEl.appendChild(div);
  });

  // Alerts
  var alerts = state.alerts || [];
  var alertEl = document.getElementById('col-alerts');
  if (alerts.length === 0) {
    alertEl.textContent = 'No alerts';
    alertEl.style.color = '#484f58';
  } else {
    alertEl.textContent = '';
    alerts.forEach(function(a) {
      var div = document.createElement('div');
      div.style.cssText = 'padding:2px 0;border-bottom:1px solid #21262d;font-size:12px;color:#f0883e';
      div.textContent = a;
      alertEl.appendChild(div);
    });
  }

  // Tile canvas
  var tiles = state.tiles;
  if (tiles && tiles.data) {
    var canvas = document.getElementById('col-tile-canvas');
    var w = tiles.w || 64; var h = tiles.h || 64;
    canvas.width = w; canvas.height = h;
    var ctx = canvas.getContext('2d');
    var img = ctx.createImageData(w, h);
    for (var i = 0; i < tiles.data.length; i++) {
      var cell = tiles.data[i];
      var color = TILE_COLORS[cell[0]] || '#30363d';
      var r2 = parseInt(color.slice(1,3)||'30',16);
      var g2 = parseInt(color.slice(3,5)||'36',16);
      var b2 = parseInt(color.slice(5,7)||'3d',16);
      var idx = i * 4;
      img.data[idx]=r2; img.data[idx+1]=g2; img.data[idx+2]=b2; img.data[idx+3]=255;
    }
    ctx.putImageData(img, 0, 0);
    document.getElementById('col-tile-pos').textContent =
      'window: (' + tiles.x + ',' + tiles.y + ') 64×64';
  }
}
```

- [ ] **Step 3: Verify visually**

Start server, open http://localhost:8181, confirm Colony tab shows cycle/resources/tile canvas with placeholder dashes when no game connected. Kill server.

- [ ] **Step 4: Commit**

```bash
git add examples/dashboard/index.html
git commit -m "feat: dashboard Colony tab — game card, resources, alerts, tile canvas"
```

---

## Task 4: Duplicants tab

**Files:**
- Modify: `examples/dashboard/index.html` — fill `#tab-duplicants` and `renderDuplicants()`

- [ ] **Step 1: Add Duplicants CSS to the style block**

Add inside the `<style>` tag:

```css
/* ── Duplicants ── */
.dupe-card {
  background: #161b22; border: 1px solid #30363d; border-radius: 4px;
  padding: 10px 12px; margin-bottom: 8px;
}
.dupe-header { display: flex; align-items: center; gap: 8px; margin-bottom:6px; }
.dupe-name { font-size: 14px; font-weight: bold; color: #c9d1d9; flex:1; }
.dupe-badge {
  font-size: 9px; padding: 1px 6px; border-radius: 2px;
  background: #21262d; color: #8b949e; border: 1px solid #30363d;
}
.dupe-badge.bionic { background: #1a2332; color: #58a6ff; border-color: #58a6ff; }
.dupe-task { font-size: 11px; color: #484f58; }
.bar-row { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
.bar-label { font-size: 10px; color: #484f58; width: 52px; flex-shrink:0; }
.bar-track { flex:1; height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 3px; }
.bar-fill.stress { background: #f85149; }
.bar-fill.health { background: #3fb950; }
.bar-fill.hunger { background: #f0883e; }
.bar-fill.bladder { background: #1f6feb; }
.bar-fill.charge { background: #a371f7; }
.bar-val { font-size: 10px; color: #8b949e; width: 32px; text-align:right; flex-shrink:0; }
.attr-grid { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.attr-chip {
  background: #21262d; border: 1px solid #30363d; border-radius: 3px;
  padding: 2px 6px; font-size: 10px; color: #8b949e;
}
.trait-list { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.trait-pill {
  background: #1a2332; border: 1px solid #30363d; border-radius: 10px;
  padding: 1px 7px; font-size: 10px; color: #58a6ff;
}
```

- [ ] **Step 2: Add Duplicants tab HTML inside `#tab-duplicants`**

Replace `<!-- Task 4 -->` with:

```html
<div id="dupe-list" style="padding:2px"></div>
```

- [ ] **Step 3: Implement `renderDuplicants()` in the script block**

Replace stub `function renderDuplicants() {}` with:

```javascript
function renderDuplicants() {
  var container = document.getElementById('dupe-list');
  if (!container) return;
  var dupes = (state && state.duplicants) || [];
  container.textContent = '';

  if (dupes.length === 0) {
    var p = document.createElement('div');
    p.style.cssText = 'color:#484f58;padding:20px;text-align:center';
    p.textContent = 'No duplicant data';
    container.appendChild(p);
    return;
  }

  dupes.forEach(function(d) {
    var card = document.createElement('div');
    card.className = 'dupe-card';

    var header = document.createElement('div');
    header.className = 'dupe-header';

    var name = document.createElement('span');
    name.className = 'dupe-name';
    name.textContent = d.name || 'Unknown';

    var type = d.type_data && d.type_data.type === 'bionic' ? 'bionic' : 'organic';
    var badge = document.createElement('span');
    badge.className = 'dupe-badge' + (type === 'bionic' ? ' bionic' : '');
    badge.textContent = type;

    var task = document.createElement('span');
    task.className = 'dupe-task';
    task.textContent = d.current_task || 'idle';

    header.appendChild(name);
    header.appendChild(badge);
    header.appendChild(task);
    card.appendChild(header);

    // Bars
    function makeBar(label, value, cls) {
      var pct = Math.min(1, Math.max(0, value || 0)) * 100;
      var row = document.createElement('div'); row.className = 'bar-row';
      var lbl = document.createElement('span'); lbl.className = 'bar-label'; lbl.textContent = label;
      var track = document.createElement('div'); track.className = 'bar-track';
      var fill = document.createElement('div'); fill.className = 'bar-fill ' + cls;
      fill.style.width = pct.toFixed(1) + '%';
      var val = document.createElement('span'); val.className = 'bar-val';
      val.textContent = (pct).toFixed(0) + '%';
      track.appendChild(fill); row.appendChild(lbl); row.appendChild(track); row.appendChild(val);
      return row;
    }

    card.appendChild(makeBar('Stress', d.stress, 'stress'));
    card.appendChild(makeBar('Health', (d.health || 0) / 100, 'health'));
    card.appendChild(makeBar('Hunger', d.hunger, 'hunger'));
    card.appendChild(makeBar('Bladder', d.bladder, 'bladder'));
    if (d.type_data && d.type_data.type === 'bionic') {
      card.appendChild(makeBar('Charge', d.type_data.charge_pct, 'charge'));
    }

    // Attributes
    var skills = d.skills || {};
    var skillKeys = Object.keys(skills);
    if (skillKeys.length > 0) {
      var grid = document.createElement('div'); grid.className = 'attr-grid';
      skillKeys.forEach(function(k) {
        var chip = document.createElement('span'); chip.className = 'attr-chip';
        chip.textContent = k + ' ' + skills[k];
        grid.appendChild(chip);
      });
      card.appendChild(grid);
    }

    // Traits
    var traits = d.traits || [];
    if (traits.length > 0) {
      var tlist = document.createElement('div'); tlist.className = 'trait-list';
      traits.forEach(function(t) {
        var pill = document.createElement('span'); pill.className = 'trait-pill';
        pill.textContent = t;
        tlist.appendChild(pill);
      });
      card.appendChild(tlist);
    }

    container.appendChild(card);
  });
}
```

- [ ] **Step 4: Commit**

```bash
git add examples/dashboard/index.html
git commit -m "feat: dashboard Duplicants tab — bars, skills, traits, bionic charge"
```

---

## Task 5: Research tab

**Files:**
- Modify: `examples/dashboard/index.html` — fill `#tab-research` and `renderResearch()`

- [ ] **Step 1: Add Research CSS to style block**

```css
/* ── Research ── */
.tech-icon {
  width: 52px; height: 52px; background: #21262d; border: 1px solid #30363d;
  border-radius: 4px; display: flex; flex-direction: column; align-items: center;
  justify-content: center; cursor: pointer; position: relative; gap: 2px;
}
.tech-icon:hover, .tech-icon.selected { border-color: #58a6ff; background: #1a2332; }
.tech-icon .ti-emoji { font-size: 20px; line-height: 1; }
.tech-icon .ti-name { font-size: 8px; color: #8b949e; text-align: center; padding: 0 2px; line-height: 1.2; }
.tech-icon .ti-check { position: absolute; top: 2px; right: 3px; color: #3fb950; font-size: 9px; }
.tech-detail {
  background: #0d1117; border: 1px solid #58a6ff; border-radius: 4px;
  padding: 8px 10px; margin-top: 6px; font-size: 11px; display: none;
}
.tech-detail.visible { display: block; }
.offer-card { background: #0d1117; border: 1px solid #30363d; border-radius: 3px; padding: 6px 8px; margin: 4px 0; }
.offer-name { color: #ffa657; font-weight: bold; font-size: 13px; margin-bottom: 3px; }
.offer-detail { color: #8b949e; font-size: 11px; }
```

- [ ] **Step 2: Add Research tab HTML inside `#tab-research`**

Replace `<!-- Task 5 -->` with:

```html
<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;height:100%">
  <div style="display:flex;flex-direction:column;gap:6px">
    <div class="card">
      <h3>Active Research</h3>
      <div class="kv"><span class="k">Tech</span><span class="v" id="res-tech" style="color:#58a6ff">—</span></div>
      <div class="progress-track"><div class="progress-fill" id="res-progress-fill" style="width:0%"></div></div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#484f58">
        <span id="res-pct">0%</span><span id="res-type"></span>
      </div>
    </div>
    <div class="card" style="flex:1">
      <h3 id="res-unlocked-header">Unlocked Technologies (0)</h3>
      <div id="res-icon-grid" style="display:flex;flex-wrap:wrap;gap:6px;padding:4px 0"></div>
      <div class="tech-detail" id="res-detail-box">
        <div style="color:#58a6ff;font-weight:bold;font-size:13px;margin-bottom:4px" id="res-detail-name"></div>
        <div style="color:#8b949e;padding:1px 0">Unlocked: <span style="color:#c9d1d9" id="res-detail-cycle"></span></div>
        <div style="color:#484f58;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin:5px 0 2px">Buildings</div>
        <div id="res-detail-buildings"></div>
      </div>
    </div>
  </div>
  <div class="card" id="pod-card">
    <h3>Printing Pod</h3>
    <div id="pod-status" style="font-size:13px;font-weight:bold;padding:4px 0 6px;color:#8b949e">—</div>
    <div class="kv"><span class="k">Timer</span><span class="v" id="pod-timer">—</span></div>
    <div id="pod-offers" style="margin-top:8px"></div>
  </div>
</div>
```

- [ ] **Step 3: Implement `renderResearch()` in the script block**

Replace stub `function renderResearch() {}` with:

```javascript
var _selectedTech = null;

function renderResearch() {
  var research = (state && state.research) || {};
  var unlocked = research.unlocked || [];
  var currentTech = research.current_tech;
  var currentProgress = research.current_progress || 0;

  setText('res-tech', currentTech || 'None');
  document.getElementById('res-progress-fill').style.width = (currentProgress * 100).toFixed(1) + '%';
  setText('res-pct', (currentProgress * 100).toFixed(0) + '%');

  // Icon grid
  document.getElementById('res-unlocked-header').textContent = 'Unlocked Technologies (' + unlocked.length + ')';
  var grid = document.getElementById('res-icon-grid');
  grid.textContent = '';
  unlocked.forEach(function(techId) {
    var icon = document.createElement('div');
    icon.className = 'tech-icon' + (_selectedTech === techId ? ' selected' : '');
    icon.setAttribute('data-tech', techId);

    var check = document.createElement('span'); check.className = 'ti-check'; check.textContent = '✓';
    var emoji = document.createElement('span'); emoji.className = 'ti-emoji'; emoji.textContent = '🔬';
    var label = document.createElement('span'); label.className = 'ti-name';
    label.textContent = techId.replace(/([A-Z])/g, ' $1').trim().slice(0, 10);

    icon.appendChild(check); icon.appendChild(emoji); icon.appendChild(label);
    icon.addEventListener('click', function() { selectTech(techId); });
    grid.appendChild(icon);
  });

  // Pod
  var pod = (state && state.printing_pod) || {};
  var podCard = document.getElementById('pod-card');
  var overdue = pod.overdue || false;
  podCard.style.borderColor = overdue ? '#f0883e' : '#30363d';
  setText('pod-status', pod.status || '—');
  document.getElementById('pod-status').style.color = overdue ? '#f0883e' : '#8b949e';
  setText('pod-timer', pod.timer || '—');

  var offersEl = document.getElementById('pod-offers');
  offersEl.textContent = '';
  (pod.offers || []).forEach(function(o) {
    var card = document.createElement('div'); card.className = 'offer-card';
    var name = document.createElement('div'); name.className = 'offer-name'; name.textContent = o.name || o;
    var detail = document.createElement('div'); detail.className = 'offer-detail'; detail.textContent = o.detail || '';
    card.appendChild(name); card.appendChild(detail);
    offersEl.appendChild(card);
  });
}

function selectTech(techId) {
  _selectedTech = techId;
  document.querySelectorAll('.tech-icon').forEach(function(el) {
    el.classList.toggle('selected', el.getAttribute('data-tech') === techId);
  });
  var box = document.getElementById('res-detail-box');
  document.getElementById('res-detail-name').textContent = techId;
  document.getElementById('res-detail-cycle').textContent = '—';
  var bld = document.getElementById('res-detail-buildings');
  bld.textContent = '';
  box.classList.add('visible');
}
```

- [ ] **Step 4: Commit**

```bash
git add examples/dashboard/index.html
git commit -m "feat: dashboard Research tab — active research, icon grid, printing pod"
```

---

## Task 6: Perimeter tab

**Files:**
- Modify: `examples/dashboard/index.html` — fill `#tab-perimeter` and `renderPerimeter()`

- [ ] **Step 1: Add Perimeter tab HTML inside `#tab-perimeter`**

Replace `<!-- Task 6 -->` with:

```html
<div style="display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto 1fr;gap:6px;height:100%">
  <div class="card" style="grid-column:1;grid-row:1/3;overflow-y:auto">
    <h3 id="peri-title">Active Perimeter</h3>
    <div class="kv"><span class="k">Goal</span><span class="v" id="peri-goal" style="color:#a371f7">—</span></div>
    <div class="kv"><span class="k">Bounds</span><span class="v" id="peri-bounds">—</span></div>
    <div class="kv"><span class="k">Blueprint</span><span class="v" id="peri-blueprint">—</span></div>
    <div class="progress-track"><div class="progress-fill" id="peri-prog-fill" style="width:0%"></div></div>
    <div style="display:flex;justify-content:space-between;font-size:11px;color:#484f58;margin-bottom:8px">
      <span id="peri-pct">0%</span><span id="peri-placed"></span>
    </div>
    <div id="peri-tasks"></div>
    <div id="peri-prereqs" style="margin-top:8px"></div>
  </div>
  <div class="card" style="grid-column:2;grid-row:1">
    <h3>Reward Tracking</h3>
    <div class="kv"><span class="k">Tick reward</span><span class="v ok" id="rew-tick">—</span></div>
    <div class="kv"><span class="k">Episode total</span><span class="v ok" id="rew-episode">—</span></div>
    <div class="kv"><span class="k">Survival layer</span><span class="v ok" id="rew-survival">—</span></div>
    <div class="kv"><span class="k">Progress layer</span><span class="v ok" id="rew-progress">—</span></div>
    <div class="kv"><span class="k">Event layer</span><span class="v" id="rew-event">—</span></div>
    <div style="margin-top:8px">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#484f58;margin-bottom:4px">Open Events</div>
      <div id="rew-open-events"></div>
    </div>
  </div>
  <div class="card" style="grid-column:2;grid-row:2;overflow-y:auto">
    <h3>Perimeter History</h3>
    <div id="peri-history"></div>
  </div>
</div>
```

- [ ] **Step 2: Implement `renderPerimeter()` in the script block**

Replace stub `function renderPerimeter() {}` with:

```javascript
function renderPerimeter() {
  var peri = (state && state.perimeter) || {};
  var rew = (state && state.reward) || {};

  setText('peri-goal', peri.goal || '—');
  setText('peri-bounds', peri.bounds ? '(' + peri.bounds[0] + ') → (' + peri.bounds[1] + ')' : '—');
  setText('peri-blueprint', peri.blueprint || '—');
  var pct = peri.progress_pct || 0;
  document.getElementById('peri-prog-fill').style.width = pct + '%';
  setText('peri-pct', pct + '% complete');
  setText('peri-placed', peri.placed != null ? peri.placed + ' / ' + peri.total + ' buildings' : '');

  // Task board
  var tasksEl = document.getElementById('peri-tasks');
  tasksEl.textContent = '';
  var sections = [
    { label: 'Completed', items: peri.tasks_done || [], state: 'done' },
    { label: 'Up Next',   items: peri.tasks_next || [], state: 'next' },
    { label: 'Blocked',   items: peri.tasks_blocked || [], state: 'blocked' },
  ];
  sections.forEach(function(s) {
    if (s.items.length === 0) return;
    var lbl = document.createElement('div');
    lbl.style.cssText = 'font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#484f58;margin:6px 0 3px';
    lbl.textContent = s.label;
    tasksEl.appendChild(lbl);
    s.items.forEach(function(t) {
      var row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:8px;padding:3px 0;border-bottom:1px solid #21262d;font-size:12px';
      var check = document.createElement('span');
      check.style.color = s.state === 'done' ? '#3fb950' : s.state === 'next' ? '#58a6ff' : '#484f58';
      check.textContent = s.state === 'done' ? '✓' : s.state === 'next' ? '▶' : '○';
      var name = document.createElement('span');
      name.style.cssText = 'flex:1' + (s.state === 'done' ? ';color:#484f58;text-decoration:line-through' : s.state === 'blocked' ? ';color:#484f58' : '');
      name.textContent = typeof t === 'string' ? t : t.name;
      row.appendChild(check); row.appendChild(name);
      if (t.dep) {
        var dep = document.createElement('span');
        dep.style.cssText = 'font-size:10px;color:#484f58';
        dep.textContent = t.dep;
        row.appendChild(dep);
      }
      tasksEl.appendChild(row);
    });
  });

  // Rewards
  function fmtR(v) { return v != null ? (v >= 0 ? '+' : '') + v.toFixed(2) : '—'; }
  setText('rew-tick', fmtR(rew.tick));
  setText('rew-episode', fmtR(rew.episode));
  setText('rew-survival', fmtR(rew.survival));
  setText('rew-progress', fmtR(rew.progress));
  setText('rew-event', fmtR(rew.event));

  var openEl = document.getElementById('rew-open-events');
  openEl.textContent = '';
  (rew.open_events || []).forEach(function(e) {
    var row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;padding:3px 0;font-size:11px;border-bottom:1px solid #21262d';
    var type = document.createElement('span'); type.style.color = '#f0883e'; type.textContent = e.type || e;
    var detail = document.createElement('span'); detail.style.cssText = 'flex:1;color:#8b949e'; detail.textContent = e.detail || '';
    var cost = document.createElement('span'); cost.style.cssText = 'font-weight:bold;color:#f85149'; cost.textContent = e.cost || '';
    row.appendChild(type); row.appendChild(detail); row.appendChild(cost);
    openEl.appendChild(row);
  });

  // History
  var histEl = document.getElementById('peri-history');
  histEl.textContent = '';
  var hist = (state && state.perimeter_history) || [];
  if (hist.length === 0) {
    var em = document.createElement('div');
    em.style.cssText = 'color:#484f58;font-style:italic;font-size:11px;padding:8px 0';
    em.textContent = 'No archived perimeters yet.';
    histEl.appendChild(em);
  } else {
    hist.forEach(function(h) {
      var row = document.createElement('div');
      row.style.cssText = 'display:flex;gap:8px;align-items:center;padding:4px 0;border-bottom:1px solid #21262d;font-size:11px';
      var goal = document.createElement('span'); goal.style.cssText = 'color:#a371f7;width:100px;flex-shrink:0'; goal.textContent = h.goal;
      var bounds = document.createElement('span'); bounds.style.cssText = 'flex:1;color:#484f58'; bounds.textContent = h.summary || '';
      var result = document.createElement('span'); result.style.cssText = 'font-weight:bold;color:' + (h.result === 'complete' ? '#3fb950' : '#484f58');
      result.textContent = h.result === 'complete' ? '✓ complete' : h.result || '—';
      row.appendChild(goal); row.appendChild(bounds); row.appendChild(result);
      histEl.appendChild(row);
    });
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add examples/dashboard/index.html
git commit -m "feat: dashboard Perimeter tab — task board, reward tracking, history"
```

---

## Task 7: Log tab

**Files:**
- Modify: `examples/dashboard/index.html` — fill `#tab-log` and `renderLog()`

- [ ] **Step 1: Add Log CSS to style block**

```css
/* ── Log ── */
.log-entry { display: flex; gap: 8px; padding: 3px 0; border-bottom: 1px solid #0d1117; font-size: 11px; align-items: baseline; }
.log-ts { color: #484f58; flex-shrink: 0; width: 70px; font-size: 10px; }
.log-badge { flex-shrink: 0; width: 60px; font-size: 10px; padding: 1px 4px; border-radius: 2px; text-align: center; }
.lb-action   { background: #1a2332; color: #58a6ff; }
.lb-ack-ok   { background: #1f3a2b; color: #3fb950; }
.lb-ack-fail { background: #2d1f1f; color: #f85149; }
.lb-reward   { background: #1f1a2e; color: #a371f7; }
.lb-agent    { background: #21262d; color: #8b949e; }
.lb-error    { background: #2d1f1f; color: #f85149; }
.lb-event    { background: #2d2210; color: #f0883e; }
.log-msg { flex: 1; color: #c9d1d9; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.log-msg.muted { color: #484f58; }
.fpill { padding: 2px 8px; border-radius: 10px; font-size: 10px; cursor: pointer; border: 1px solid #30363d; background: #21262d; color: #8b949e; }
.fpill.on { border-color: #58a6ff; color: #58a6ff; background: #1a2332; }
```

- [ ] **Step 2: Add Log tab HTML inside `#tab-log`**

Replace `<!-- Task 7 -->` with:

```html
<div style="display:grid;grid-template-columns:1fr 200px;gap:6px;height:100%">
  <div class="card" style="display:flex;flex-direction:column">
    <h3>Session Feed</h3>
    <div id="log-filters" style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px">
      <span class="fpill on" data-filter="all" onclick="toggleFilter(this)">All</span>
      <span class="fpill on" data-filter="action" onclick="toggleFilter(this)">Actions</span>
      <span class="fpill on" data-filter="ack" onclick="toggleFilter(this)">ACKs</span>
      <span class="fpill on" data-filter="reward" onclick="toggleFilter(this)">Rewards</span>
      <span class="fpill on" data-filter="event" onclick="toggleFilter(this)">Events</span>
      <span class="fpill on" data-filter="agent" onclick="toggleFilter(this)">Agent</span>
      <span class="fpill on" data-filter="error" onclick="toggleFilter(this)">Errors</span>
    </div>
    <div id="log-feed" style="overflow-y:auto;flex:1"></div>
  </div>
  <div style="display:flex;flex-direction:column;gap:6px">
    <div class="card">
      <h3>Session Stats</h3>
      <div class="kv"><span class="k">Ticks</span><span class="v" id="ls-ticks">0</span></div>
      <div class="kv"><span class="k">Actions</span><span class="v" id="ls-actions">0</span></div>
      <div class="kv"><span class="k">ACK ok</span><span class="v ok" id="ls-ack-ok">0</span></div>
      <div class="kv"><span class="k">ACK fail</span><span class="v warn" id="ls-ack-fail">0</span></div>
      <div class="kv"><span class="k">No-ops</span><span class="v" id="ls-noops">0</span></div>
      <div class="kv"><span class="k">Errors</span><span class="v bad" id="ls-errors">0</span></div>
    </div>
    <div class="card">
      <h3>Episode</h3>
      <div class="kv"><span class="k">Phase</span><span class="v" id="ls-phase">—</span></div>
      <div class="kv"><span class="k">Cycle</span><span class="v" id="ls-cycle">—</span></div>
      <div class="kv"><span class="k">Reward</span><span class="v ok" id="ls-reward">—</span></div>
      <div class="kv"><span class="k">Deaths</span><span class="v ok" id="ls-deaths">0</span></div>
    </div>
  </div>
</div>
```

- [ ] **Step 3: Implement `renderLog()` and filter logic in the script block**

Replace stub `function renderLog() {}` with:

```javascript
var _logFilters = { all: true, action: true, ack: true, reward: true, event: true, agent: true, error: true };

function toggleFilter(el) {
  var f = el.getAttribute('data-filter');
  _logFilters[f] = !_logFilters[f];
  el.classList.toggle('on', _logFilters[f]);
  renderLog();
}

function renderLog() {
  // Stats
  setText('ls-ticks', sessionStats.ticks);
  setText('ls-actions', sessionStats.actions);
  setText('ls-ack-ok', sessionStats.ackOk);
  setText('ls-ack-fail', sessionStats.ackFail);
  setText('ls-noops', sessionStats.noops);
  setText('ls-errors', sessionStats.errors);
  setText('ls-phase', state.phase != null ? state.phase : '—');
  setText('ls-cycle', state.cycle != null ? state.cycle + ' / 100' : '—');
  var rew = (state && state.reward) || {};
  setText('ls-reward', rew.episode != null ? (rew.episode >= 0 ? '+' : '') + rew.episode.toFixed(1) : '—');

  // Feed
  var feed = document.getElementById('log-feed');
  if (!feed) return;
  var wasAtBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 40;
  feed.textContent = '';

  var BADGE_CLASS = {
    'action': 'lb-action', 'ack-ok': 'lb-ack-ok', 'ack-fail': 'lb-ack-fail',
    'reward': 'lb-reward', 'agent': 'lb-agent', 'error': 'lb-error', 'event': 'lb-event',
  };
  var BADGE_LABEL = {
    'action': 'action', 'ack-ok': 'ack ✓', 'ack-fail': 'ack ✗',
    'reward': 'reward', 'agent': 'agent', 'error': 'error', 'event': 'event',
  };

  var shown = logEntries.filter(function(e) {
    if (_logFilters.all) return true;
    var cat = e.type.startsWith('ack') ? 'ack' : e.type;
    return _logFilters[cat];
  });

  shown.slice(-200).forEach(function(e) {
    var row = document.createElement('div'); row.className = 'log-entry';
    var ts = document.createElement('span'); ts.className = 'log-ts'; ts.textContent = e.ts;
    var badge = document.createElement('span');
    badge.className = 'log-badge ' + (BADGE_CLASS[e.type] || 'lb-agent');
    badge.textContent = BADGE_LABEL[e.type] || e.type;
    var msg = document.createElement('span');
    msg.className = 'log-msg' + (e.type === 'agent' && e.msg && e.msg.includes('no_op') ? ' muted' : '');
    msg.textContent = e.msg;
    row.appendChild(ts); row.appendChild(badge); row.appendChild(msg);
    feed.appendChild(row);
  });

  if (wasAtBottom) feed.scrollTop = feed.scrollHeight;
}
```

- [ ] **Step 4: Commit**

```bash
git add examples/dashboard/index.html
git commit -m "feat: dashboard Log tab — feed with type filters, session stats, episode stats"
```

---

## Task 8: Config tab

**Files:**
- Modify: `examples/dashboard/index.html` — fill `#tab-config`, `renderConfig()`, `loadConfig()`

- [ ] **Step 1: Add Config CSS to style block**

```css
/* ── Config ── */
.profile-row {
  display: flex; align-items: center; gap: 8px; padding: 6px 8px;
  background: #0d1117; border: 1px solid #30363d; border-radius: 4px;
  cursor: pointer; margin-bottom: 4px;
}
.profile-row:hover { border-color: #484f58; }
.profile-row.selected { border-color: #58a6ff; background: #1a2332; }
.profile-dot { width: 8px; height: 8px; border-radius: 50%; border: 2px solid #484f58; flex-shrink: 0; }
.profile-dot.active { border-color: #3fb950; background: #3fb950; }
.profile-info { flex: 1; display: flex; flex-direction: column; gap: 1px; }
.profile-name-text { font-size: 12px; color: #c9d1d9; }
.profile-sub { font-size: 10px; color: #484f58; }
.profile-active-badge {
  font-size: 9px; color: #3fb950; background: #1f3a2b;
  border: 1px solid #2ea043; padding: 1px 5px; border-radius: 2px;
}
.cfg-field { display: flex; flex-direction: column; gap: 3px; margin-bottom: 7px; }
.cfg-field label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #484f58; }
.cfg-field input, .cfg-field select {
  background: #0d1117; border: 1px solid #30363d; color: #c9d1d9;
  padding: 5px 8px; border-radius: 3px; font-family: inherit; font-size: 12px; outline: none;
}
.cfg-field input:focus { border-color: #58a6ff; }
.conn-badge {
  display: inline-flex; align-items: center; gap: 5px; padding: 2px 8px;
  border-radius: 10px; font-size: 10px; margin-bottom: 8px;
}
.conn-badge.ok { background: #1f3a2b; color: #3fb950; border: 1px solid #2ea043; }
.conn-badge.err { background: #2d1f1f; color: #f85149; border: 1px solid #f85149; }
.conn-badge.idle { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
.cfg-btn {
  padding: 4px 12px; border-radius: 3px; cursor: pointer;
  font-family: inherit; font-size: 11px; border: 1px solid;
}
.cfg-btn.primary { background: #1a2332; border-color: #58a6ff; color: #58a6ff; }
.cfg-btn.ok { background: #1f3a2b; border-color: #2ea043; color: #3fb950; }
.cfg-btn.danger { background: #2d1f1f; border-color: #f85149; color: #f85149; }
.cfg-btn.neutral { background: #21262d; border-color: #30363d; color: #8b949e; }
.cfg-note { font-size: 10px; color: #484f58; font-style: italic; margin-top: 4px; }
.cfg-divider { border-top: 1px solid #21262d; margin: 8px 0; }
```

- [ ] **Step 2: Add Config tab HTML inside `#tab-config`**

Replace `<!-- Task 8 -->` with:

```html
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;height:100%;align-content:start">
  <!-- Left: LLM profiles -->
  <div class="card">
    <h3>LLM Backend Profiles</h3>
    <div id="cfg-profile-list"></div>
    <button class="cfg-btn neutral" style="margin:6px 0;width:100%" onclick="cfgAddProfile()">+ Add Profile</button>
    <div class="cfg-divider"></div>
    <div id="cfg-edit-section" style="display:none">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#58a6ff;margin-bottom:6px" id="cfg-edit-label">Edit Profile</div>
      <div class="cfg-field"><label>Profile Name</label><input type="text" id="cfg-f-name"></div>
      <div class="cfg-field"><label>Endpoint URL</label><input type="text" id="cfg-f-url" placeholder="http://host:port/v1"></div>
      <div class="cfg-field"><label>Model</label><input type="text" id="cfg-f-model" placeholder="org/model-name"></div>
      <div class="cfg-field"><label>API Key (optional)</label><input type="text" id="cfg-f-key" placeholder="leave blank if not required"></div>
      <div class="cfg-field">
        <label>Vision Pipeline</label>
        <select id="cfg-f-vision">
          <option value="false">Disabled — text state only</option>
          <option value="true">Enabled — attach screenshot each tick</option>
        </select>
      </div>
      <div style="display:flex;gap:6px;margin-top:8px;align-items:center">
        <button class="cfg-btn ok" onclick="cfgSetActive()">Set Active</button>
        <button class="cfg-btn primary" onclick="cfgSaveProfile()">Save</button>
        <button class="cfg-btn danger" style="margin-left:auto" onclick="cfgDeleteProfile()">Delete</button>
      </div>
      <div id="cfg-conn-badge" class="conn-badge idle" style="margin-top:8px">● Not tested</div>
    </div>
  </div>

  <!-- Right: connections -->
  <div style="display:flex;flex-direction:column;gap:10px">
    <div class="card">
      <h3>Game Connection (ONIBridge TCP)</h3>
      <div id="cfg-game-badge" class="conn-badge idle">● Unknown</div>
      <div class="cfg-field"><label>Host</label><input type="text" id="cfg-game-host" value="10.0.0.10"></div>
      <div class="cfg-field"><label>Port</label><input type="text" id="cfg-game-port" value="9999"></div>
      <div style="display:flex;gap:6px;margin-top:8px">
        <button class="cfg-btn ok" onclick="cfgApplyGame()">Apply</button>
        <button class="cfg-btn neutral" onclick="cfgTestGame()">Test</button>
      </div>
    </div>
    <div class="card">
      <h3>Runner Relay (WebSocket)</h3>
      <div id="cfg-relay-badge" class="conn-badge idle">● Unknown</div>
      <div class="cfg-field"><label>Relay Port</label><input type="text" id="cfg-relay-port" value="8182"></div>
      <p class="cfg-note">Port change takes effect on next runner start.</p>
    </div>
    <div class="card">
      <h3>Runner Process</h3>
      <div id="cfg-runner-badge" class="conn-badge idle">● Stopped</div>
      <div style="display:flex;gap:6px;margin-top:8px">
        <button class="cfg-btn neutral" onclick="cfgRestartRunner()">Restart</button>
        <button class="cfg-btn danger" onclick="cfgStopRunner()">Stop</button>
      </div>
      <p class="cfg-note">Restart picks up all config changes immediately.</p>
    </div>
  </div>
</div>
```

- [ ] **Step 3: Implement config tab JS in the script block**

Add after the existing script functions:

```javascript
var _cfgProfiles = [];
var _cfgActiveId = null;
var _cfgSelectedId = null;

function loadConfig() {
  fetch('/config/profiles').then(function(r) { return r.json(); }).then(function(data) {
    _cfgProfiles = data.profiles || [];
    _cfgActiveId = data.active_id;
    renderProfileList();
    if (_cfgProfiles.length > 0) selectProfile(_cfgProfiles[0].id);
  });
  fetch('/config/game').then(function(r) { return r.json(); }).then(function(data) {
    if (document.getElementById('cfg-game-host')) {
      document.getElementById('cfg-game-host').value = data.host || '';
      document.getElementById('cfg-game-port').value = data.port || '';
    }
  });
  fetch('/runner/status').then(function(r) { return r.json(); }).then(function(data) {
    updateRunnerBadge(data);
  });
}

function renderConfig() {
  var badge = document.getElementById('cfg-runner-badge');
  if (!badge) return;
  fetch('/runner/status').then(function(r) { return r.json(); }).then(function(data) {
    updateRunnerBadge(data);
  });
}

function updateRunnerBadge(data) {
  var badge = document.getElementById('cfg-runner-badge');
  if (!badge) return;
  if (data.running) {
    var uptime = data.uptime_seconds != null ? ' · uptime ' + fmtUptime(data.uptime_seconds) : '';
    badge.className = 'conn-badge ok';
    badge.textContent = '● Running — pid ' + data.pid + uptime;
  } else {
    badge.className = 'conn-badge idle';
    badge.textContent = '● Stopped';
  }
}

function fmtUptime(sec) {
  var h = Math.floor(sec / 3600); var m = Math.floor((sec % 3600) / 60); var s = sec % 60;
  return String(h).padStart(2,'0') + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
}

function renderProfileList() {
  var list = document.getElementById('cfg-profile-list');
  if (!list) return;
  list.textContent = '';
  _cfgProfiles.forEach(function(p) {
    var row = document.createElement('div');
    row.className = 'profile-row' + (_cfgSelectedId === p.id ? ' selected' : '');
    row.setAttribute('data-id', p.id);

    var dot = document.createElement('span');
    dot.className = 'profile-dot' + (p.id === _cfgActiveId ? ' active' : '');

    var info = document.createElement('div'); info.className = 'profile-info';
    var nm = document.createElement('span'); nm.className = 'profile-name-text'; nm.textContent = p.name;
    var sub = document.createElement('span'); sub.className = 'profile-sub';
    sub.textContent = (p.endpoint_url || '').replace('http://', '').replace('https://', '').split('/')[0] + ' · ' + (p.model || '');
    info.appendChild(nm); info.appendChild(sub);

    row.appendChild(dot); row.appendChild(info);
    if (p.id === _cfgActiveId) {
      var badge = document.createElement('span'); badge.className = 'profile-active-badge'; badge.textContent = 'Active';
      row.appendChild(badge);
    }
    row.addEventListener('click', function() { selectProfile(p.id); });
    list.appendChild(row);
  });
}

function selectProfile(id) {
  _cfgSelectedId = id;
  var p = _cfgProfiles.find(function(x) { return x.id === id; });
  if (!p) return;
  renderProfileList();
  document.getElementById('cfg-edit-section').style.display = 'block';
  document.getElementById('cfg-edit-label').textContent = 'Edit: ' + p.name;
  document.getElementById('cfg-f-name').value = p.name || '';
  document.getElementById('cfg-f-url').value = p.endpoint_url || '';
  document.getElementById('cfg-f-model').value = p.model || '';
  document.getElementById('cfg-f-key').value = p.api_key || '';
  document.getElementById('cfg-f-vision').value = p.vision_enabled ? 'true' : 'false';
}

function cfgSaveProfile() {
  if (!_cfgSelectedId) return;
  var updates = {
    name: document.getElementById('cfg-f-name').value,
    endpoint_url: document.getElementById('cfg-f-url').value,
    model: document.getElementById('cfg-f-model').value,
    api_key: document.getElementById('cfg-f-key').value,
    vision_enabled: document.getElementById('cfg-f-vision').value === 'true',
  };
  fetch('/config/profiles/' + _cfgSelectedId, {
    method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(updates)
  }).then(function() { loadConfig(); });
}

function cfgSetActive() {
  if (!_cfgSelectedId) return;
  fetch('/config/profiles/' + _cfgSelectedId + '/activate', { method: 'POST' })
    .then(function() { loadConfig(); });
}

function cfgDeleteProfile() {
  if (!_cfgSelectedId) return;
  if (!confirm('Delete this profile?')) return;
  fetch('/config/profiles/' + _cfgSelectedId, { method: 'DELETE' })
    .then(function(r) {
      if (r.ok) { _cfgSelectedId = null; document.getElementById('cfg-edit-section').style.display = 'none'; loadConfig(); }
      else { r.json().then(function(d) { alert(d.error || 'Cannot delete'); }); }
    });
}

function cfgAddProfile() {
  fetch('/config/profiles', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ name: 'New Profile', endpoint_url: '', model: '', api_key: '', vision_enabled: false })
  }).then(function(r) { return r.json(); }).then(function(d) {
    loadConfig();
    setTimeout(function() { selectProfile(d.id); }, 200);
  });
}

function cfgApplyGame() {
  var host = document.getElementById('cfg-game-host').value;
  var port = parseInt(document.getElementById('cfg-game-port').value, 10);
  fetch('/config/game', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ host: host, port: port })
  }).then(function() {
    var badge = document.getElementById('cfg-game-badge');
    badge.className = 'conn-badge ok'; badge.textContent = '● Applied';
  });
}

function cfgTestGame() {
  var badge = document.getElementById('cfg-game-badge');
  badge.className = 'conn-badge idle'; badge.textContent = '● Testing...';
  fetch('/runner/status').then(function(r) { return r.json(); }).then(function(d) {
    badge.className = d.running ? 'conn-badge ok' : 'conn-badge idle';
    badge.textContent = d.running ? '● Runner connected' : '● Runner not running';
  });
}

function cfgRestartRunner() {
  fetch('/runner/stop', { method: 'POST' }).then(function() {
    return fetch('/runner/start', { method: 'POST' });
  }).then(function() { renderConfig(); });
}

function cfgStopRunner() {
  fetch('/runner/stop', { method: 'POST' }).then(function() { renderConfig(); });
}
```

- [ ] **Step 4: Commit**

```bash
git add examples/dashboard/index.html
git commit -m "feat: dashboard Config tab — LLM profiles CRUD, game config, runner controls"
```

---

## Task 9: Wire reward/obligation data into health strip

**Files:**
- Modify: `examples/dashboard/server.py` — track reward state server-side
- Modify: `examples/dashboard/index.html` — parse reward/obligation fields from state

The health strip needs `reward.tick`, `reward.episode`, `survival`, and `obligations`. These come from the runner relay. This task ensures `handleMessage` populates `rewardData` correctly.

- [ ] **Step 1: Update `handleMessage` in index.html to extract reward data**

In the `handleMessage` function, inside the `if (msg.type === 'state')` branch, add after `state = msg.data || {};`:

```javascript
    // Extract reward and obligation data for health strip
    var r = state.reward || {};
    rewardData.tick = r.tick != null ? r.tick : null;
    rewardData.episode = r.episode != null ? r.episode : 0;
    var alive = (state.duplicants || []).filter(function(d) { return d.health > 0; }).length;
    var total = (state.duplicants || []).length;
    rewardData.survival = total > 0 ? alive + '/' + total : null;
    rewardData.obligations = r.open_events ? r.open_events.map(function(e) {
      return (e.type || e) + (e.detail ? ': ' + e.detail : '');
    }) : [];
```

- [ ] **Step 2: Verify health strip updates when game is connected**

Start dashboard, connect to a running game (or send a mock state via the relay). Confirm health strip shows non-dash values.

- [ ] **Step 3: Commit**

```bash
git add examples/dashboard/index.html
git commit -m "feat: wire reward and obligation data into colony health strip"
```

---

## Task 10: Remove old Gemini-specific runner start logic from server.py

**Files:**
- Modify: `examples/dashboard/server.py`

The `start_runner()` function currently requires `GOOGLE_API_KEY` and hardcodes the `--api-key` flag. This needs to read from the active LLM profile instead.

- [ ] **Step 1: Write failing test**

Add to `tests/dashboard/test_server.py`:

```python
def test_runner_start_uses_active_profile():
    """start_runner() should not require GOOGLE_API_KEY env var."""
    import examples.dashboard.server as srv
    # Ensure no google key set
    old = os.environ.pop("GOOGLE_API_KEY", None)
    try:
        # Should not return an error about missing key
        # (will fail on actual subprocess but not on key check)
        result = srv._build_runner_cmd()
        assert "--api-key" not in result or result[result.index("--api-key") + 1] != ""
    finally:
        if old:
            os.environ["GOOGLE_API_KEY"] = old
```

- [ ] **Step 2: Refactor `start_runner()` to use active profile**

In `server.py`, replace `start_runner()` with:

```python
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
    ]
    if profile:
        cmd += ["--endpoint", profile.get("endpoint_url", "")]
        cmd += ["--model", profile.get("model", "")]
        if profile.get("api_key"):
            cmd += ["--api-key", profile["api_key"]]
        if profile.get("vision_enabled"):
            cmd += ["--vision"]
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
```

Also remove the `GOOGLE_API_KEY` global at the top of `server.py` — it's no longer used.

- [ ] **Step 3: Run all server tests**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
pytest tests/dashboard/test_server.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add examples/dashboard/server.py tests/dashboard/test_server.py
git commit -m "refactor: runner start reads active LLM profile instead of GOOGLE_API_KEY env"
```

---

## Task 11: Manual smoke test and cleanup

- [ ] **Step 1: Start the full stack**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
python3 examples/dashboard/server.py
```

Open http://localhost:8181 in browser.

- [ ] **Step 2: Verify each tab**

- Colony tab: dashes show, no JS errors in console
- Duplicants tab: empty list with placeholder message
- Research tab: dashes, pod card present
- Perimeter tab: empty task board, history shows "No archived perimeters yet"
- Log tab: empty feed, filter pills toggle on/off, stats show zeroes
- Config tab: profile list loads from `llm_profiles.json`, add/save/delete work, game config fields editable, runner badge updates

- [ ] **Step 3: Connect to live game if available**

If ONI + ONIBridge is running, start the runner from Config tab and verify all tabs populate with real data.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: dashboard redesign complete — 6 tabs, LLM profiles, health strip"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered by |
|---|---|
| Status bar | Task 2 |
| Tab bar (6 tabs) | Task 2 |
| Colony tab | Task 3 |
| Duplicants tab (skills/traits/hunger/bladder/bionic) | Task 4 |
| Research tab (icon grid, printing pod) | Task 5 |
| Perimeter tab (task board, reward, history) | Task 6 |
| Log tab (filters, stats sidebar) | Task 7 |
| Config tab (LLM profiles CRUD, game/relay/runner) | Task 8 |
| Colony Health strip | Task 2 (shell) + Task 9 (wired) |
| LLM profiles persisted to JSON | Task 1 |
| Config API endpoints (GET/POST/PUT/DELETE/activate) | Task 1 |
| Game config endpoint | Task 1 |
| Runner uptime tracking | Task 1 |
| Runner uses active profile (not GOOGLE_API_KEY) | Task 10 |
| Vision pipeline toggle per profile | Task 8 |
| llm_profiles.json gitignored | Task 1 |

All spec requirements covered.
