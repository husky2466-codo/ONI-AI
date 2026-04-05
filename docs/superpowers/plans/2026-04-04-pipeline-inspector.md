# Pipeline Inspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live Pipeline tab to the ONI dashboard that shows every stage of the inference pipeline per tick — formatted prompt, raw LLM response, validation decisions, action sent, and game ack — so the operator can debug without touching code or log files.

**Architecture:** `llm.py` returns a `(action, prompt_text, raw_response)` tuple instead of just the action dict. `runner.py` collects per-stage data into a `PipelineSnapshot` each tick and broadcasts it as a `{"type": "pipeline", ...}` WebSocket message. `server.py` stores the last 50 snapshots and sends the last 10 to new browser clients. A new Pipeline tab in `index.html` renders a tick feed on the left and an expandable stage detail panel on the right.

**Tech Stack:** Python 3.14, FastAPI, asyncio, vanilla JavaScript, existing WebSocket infrastructure (no new dependencies)

**Security note:** The Pipeline tab renders content from the local Python runner via innerHTML. All interpolated values are passed through `escHtml()` before insertion — this is the correct XSS mitigation for a local-only dashboard. No external user input reaches the DOM.

---

## File Map

| File | Change |
|------|--------|
| `src/agent/llm.py` | `decide()` returns `tuple[dict, str, str]` — (action, prompt_text, raw_response) |
| `src/agent/runner.py` | Add `PipelineSnapshot` dataclass, collect per-stage data, broadcast `pipeline` message |
| `examples/dashboard/server.py` | Store last 50 pipeline snapshots, send last 10 on connect, broadcast to browsers |
| `examples/dashboard/index.html` | Add Pipeline tab: tick feed + stage detail panel |
| `tests/agent/test_llm_pipeline.py` | New test file for tuple return and snapshot structure |
| `tests/agent/test_pipeline_snapshot.py` | New test file for PipelineSnapshot class |

---

## Task 1: Change llm.py decide() to return a tuple

**Files:**
- Modify: `src/agent/llm.py:354-399`
- Test: `tests/agent/test_llm_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_llm_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agent/test_llm_pipeline.py -v
```

Expected: 4 failures — `decide()` currently returns a `dict`, not a `tuple`.

- [ ] **Step 3: Update decide() to return a tuple**

In `src/agent/llm.py`, replace the `decide()` method (lines 354-399) with:

```python
def decide(
    self,
    state_data: dict[str, Any],
    pending_action: "dict | None" = None,
    ledger_context: str = "",
    colony_health: str = "",
    last_ack: "dict | None" = None,
) -> "tuple[dict[str, Any], str, str]":
    """
    Given a state snapshot dict, return (action_dict, prompt_text, raw_llm_response).
    Falls back to (no_op, prompt, "") on any failure.
    """
    prompt = _format_state(
        state_data,
        pending_action=pending_action,
        ledger_context=ledger_context,
        colony_health=colony_health,
        last_ack=last_ack,
    )

    try:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )

        usage = response.usage
        if usage:
            self.total_input_tokens  += usage.prompt_tokens or 0
            self.total_output_tokens += usage.completion_tokens or 0
            self.total_calls         += 1

        raw = (response.choices[0].message.content or "").strip()
        logger.info("LLM prompt:\n%s", prompt)
        logger.info("LLM raw response: %s", raw)
        return self._parse_action(raw), prompt, raw

    except Exception as e:
        logger.warning("LLM call failed: %s -- sending no_op", e)
        return build_no_op(), prompt, ""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agent/test_llm_pipeline.py -v
```

Expected: 4 PASS

- [ ] **Step 5: Run full agent test suite**

```bash
pytest tests/agent/ -v
```

Expected: all pass. The existing runner tests do not call `agent.decide()` directly so no other callers break yet.

- [ ] **Step 6: Commit**

```bash
git add src/agent/llm.py tests/agent/test_llm_pipeline.py
git commit -m "feat: decide() returns (action, prompt, raw_response) tuple for pipeline inspector"
```

---

## Task 2: Add PipelineSnapshot to runner.py and broadcast pipeline message

**Files:**
- Modify: `src/agent/runner.py`
- Test: `tests/agent/test_pipeline_snapshot.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_pipeline_snapshot.py`:

```python
# tests/agent/test_pipeline_snapshot.py
from src.agent.runner import PipelineSnapshot


def test_pipeline_snapshot_to_dict_has_required_keys():
    snap = PipelineSnapshot(tick=5, cycle=2)
    snap.add_stage("state_in", "State Received", {"cycle": 2, "dupes": 3})
    snap.add_stage("prompt", "Prompt Formatted", {"chars": 500, "preview": "Cycle: 2"})
    snap.elapsed_ms = 8200
    d = snap.to_dict()
    assert d["type"] == "pipeline"
    assert d["tick"] == 5
    assert d["cycle"] == 2
    assert d["elapsed_ms"] == 8200
    assert isinstance(d["stages"], list)
    assert len(d["stages"]) == 2


def test_pipeline_snapshot_stage_structure():
    snap = PipelineSnapshot(tick=1, cycle=1)
    snap.add_stage("validation", "Validation", {"result": "blocked", "reason": "not solid"})
    stage = snap.to_dict()["stages"][0]
    assert stage["name"] == "validation"
    assert stage["label"] == "Validation"
    assert stage["data"]["result"] == "blocked"


def test_pipeline_snapshot_empty_stages():
    snap = PipelineSnapshot(tick=1, cycle=1)
    d = snap.to_dict()
    assert d["stages"] == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agent/test_pipeline_snapshot.py -v
```

Expected: 3 failures — `PipelineSnapshot` does not exist yet.

- [ ] **Step 3: Add PipelineSnapshot class to runner.py**

Add `from dataclasses import dataclass, field` and `import time as _time` to the imports at the top of `src/agent/runner.py`.

Then add this class after the `_validate_dig` function and before `async def run(`:

```python
@dataclass
class PipelineSnapshot:
    """Captures per-stage data for one agent tick. Broadcast to dashboard as pipeline message."""
    tick: int
    cycle: int
    elapsed_ms: int = 0
    _stages: list = field(default_factory=list, repr=False)

    def add_stage(self, name: str, label: str, data: dict) -> None:
        self._stages.append({"name": name, "label": label, "data": data})

    def to_dict(self) -> dict:
        return {
            "type": "pipeline",
            "tick": self.tick,
            "cycle": self.cycle,
            "elapsed_ms": self.elapsed_ms,
            "stages": list(self._stages),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agent/test_pipeline_snapshot.py -v
```

Expected: 3 PASS

- [ ] **Step 5: Wire PipelineSnapshot into the tick loop**

In `runner.py` inside `async def run()`, find the block that starts with:
```python
            colony_health = format_colony_health(
```

Replace everything from that line through `logger.info("  -> AI action: %s", action)` with the following. This fixes the tuple unpack AND collects all pipeline stages:

```python
            colony_health = format_colony_health(
                state.data, tick_reward, episode_reward, obligations
            )

            snap = PipelineSnapshot(tick=tick, cycle=cycle)
            snap_start = _time.monotonic()

            # Stage: state_in
            _dups = state.data.get("duplicants", [])
            _tiles = state.data.get("tiles", {})
            _tw = f"{_tiles.get('w',0)}x{_tiles.get('h',0)} @ ({_tiles.get('x',0)},{_tiles.get('y',0)})"
            snap.add_stage("state_in", "State Received", {
                "cycle": cycle,
                "dupes": len(_dups),
                "o2_kg": round(state.data.get("resources", {}).get("oxygen_kg", 0), 2),
                "alerts": len(state.data.get("alerts", [])),
                "tile_window": _tw,
            })

            loop = asyncio.get_event_loop()
            last_ack_dict = (
                {"action": client.last_ack.action, "success": client.last_ack.success,
                 "error": client.last_ack.error}
                if client.last_ack else None
            )

            _llm_start = _time.monotonic()
            candidate_tuple = await loop.run_in_executor(
                _llm_executor,
                lambda: agent.decide(
                    state.data,
                    pending_action=relay.pending_action,
                    ledger_context=ledger.format_context(),
                    colony_health=colony_health,
                    last_ack=last_ack_dict,
                ),
            )
            _llm_elapsed_ms = int((_time.monotonic() - _llm_start) * 1000)
            candidate, prompt_text, raw_response = candidate_tuple

            # Stage: prompt
            snap.add_stage("prompt", "Prompt Formatted", {
                "chars": len(prompt_text),
                "tokens_est": len(prompt_text) // 4,
                "preview": prompt_text[:500],
            })

            # Stage: llm_call
            snap.add_stage("llm_call", "LLM Response", {
                "model": agent._model,
                "elapsed_ms": _llm_elapsed_ms,
                "raw_response": raw_response[:800],
                "extracted_json": candidate,
            })

            # Validation chain
            _validation_input = dict(candidate)
            _validation_result = "passed"
            _validation_reason = ""

            if candidate.get("action") == "place_perimeter" and ledger.active is not None:
                logger.info("  -> runner blocked place_perimeter (perimeter already active)")
                _validation_result = "blocked"
                _validation_reason = "perimeter already active"
                candidate = build_no_op()

            if candidate.get("action") == "dig":
                _validated = _validate_dig(candidate, state.data)
                if _validated.get("action") == "no_op":
                    _validation_result = "blocked"
                    _validation_reason = f"cell ({candidate.get('cell_x')},{candidate.get('cell_y')}) is not solid"
                    candidate = _validated

            if (candidate == last_ai_action
                    and candidate.get("action") != "no_op"
                    and (tick - last_ai_action_tick) < _DEDUP_TTL):
                logger.info("  -> dedup: suppressing repeat %s (sent %d ticks ago)",
                            candidate.get("action"), tick - last_ai_action_tick)
                _validation_result = "deduped"
                _validation_reason = f"same as tick {last_ai_action_tick}"
                action = build_no_op()
            else:
                action = candidate
                last_ai_action = candidate
                last_ai_action_tick = tick

            # Stage: validation
            snap.add_stage("validation", "Validation", {
                "input_action": _validation_input,
                "result": _validation_result,
                "reason": _validation_reason,
                "output_action": action,
            })

            logger.info("  -> AI action: %s", action)
```

- [ ] **Step 6: Add sent and ack stages, finalize and broadcast snapshot**

After `await client.send_action(action)` and `relay.pending_action = action`, add:

```python
            # Stage: sent
            snap.add_stage("sent", "Sent to Game", {"action": action})

            # Stage: ack (reflects ack from previous action — arrives next tick)
            if client.last_ack:
                snap.add_stage("ack", "Game ACK", {
                    "action": client.last_ack.action,
                    "success": client.last_ack.success,
                    "error": client.last_ack.error,
                })

            snap.elapsed_ms = int((_time.monotonic() - snap_start) * 1000)
```

Then after the existing `await relay.broadcast({"type": "ack", ...})` call, add:

```python
            await relay.broadcast(snap.to_dict())
```

- [ ] **Step 7: Run full test suite**

```bash
pytest tests/agent/ -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/agent/runner.py tests/agent/test_pipeline_snapshot.py
git commit -m "feat: add PipelineSnapshot to runner, broadcast pipeline message per tick"
```

---

## Task 3: Store and relay pipeline snapshots in server.py

**Files:**
- Modify: `examples/dashboard/server.py:180-184` (storage declarations)
- Modify: `examples/dashboard/server.py:222-227` (relay_loop message handling)
- Modify: `examples/dashboard/server.py:276-278` (websocket_endpoint on-connect send)

- [ ] **Step 1: Add pipeline_snapshots storage**

In `server.py`, add alongside the existing storage declarations (around line 180):

```python
pipeline_snapshots: list[dict] = []
```

- [ ] **Step 2: Append incoming pipeline messages in relay_loop**

In `relay_loop()` message handling block, add the pipeline case:

```python
                    if msg.get("type") == "state":
                        last_state = msg.get("data", {})
                    elif msg.get("type") == "ack":
                        log_entries = msg.get("log", log_entries)
                    elif msg.get("type") == "pipeline":
                        pipeline_snapshots.append(msg)
                        if len(pipeline_snapshots) > 50:
                            pipeline_snapshots.pop(0)
                    await broadcast_browsers(msg)
```

- [ ] **Step 3: Send last 10 snapshots on browser connect**

In `websocket_endpoint`, after the existing `if log_entries:` block, add:

```python
    if pipeline_snapshots:
        for snap in pipeline_snapshots[-10:]:
            await ws.send_json(snap)
```

- [ ] **Step 4: Verify server starts cleanly**

```bash
python3 examples/dashboard/server.py
```

Expected: starts on :8181, no errors. Stop with Ctrl+C.

- [ ] **Step 5: Commit**

```bash
git add examples/dashboard/server.py
git commit -m "feat: server.py stores and relays pipeline snapshots to browser clients"
```

---

## Task 4: Add Pipeline tab to index.html

**Files:**
- Modify: `examples/dashboard/index.html`

All innerHTML assignments use escHtml() for string interpolation — this is the correct XSS mitigation for trusted local content.

- [ ] **Step 1: Add tab button**

Find in `index.html`:
```html
  <button data-tab="log" onclick="switchTab('log')">📋 Log</button>
  <button data-tab="config" onclick="switchTab('config')">⚙️ Config</button>
```

Replace with:
```html
  <button data-tab="log" onclick="switchTab('log')">📋 Log</button>
  <button data-tab="pipeline" onclick="switchTab('pipeline')">🔬 Pipeline</button>
  <button data-tab="config" onclick="switchTab('config')">⚙️ Config</button>
```

- [ ] **Step 2: Add tab body HTML**

After the closing `</div></div>` of the log tab body (the line containing `</div></div>` that ends the log tab), insert:

```html
  <div class="tab-body" id="tab-pipeline"><div style="display:grid;grid-template-columns:260px 1fr;gap:6px;height:100%">
  <div class="card" style="display:flex;flex-direction:column;overflow:hidden">
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
      <h3 style="margin:0;flex:1">Tick Feed</h3>
      <span id="pl-auto-label" style="font-size:10px;color:#3fb950">AUTO</span>
      <button class="tb-btn" onclick="plToggleAuto()" id="pl-auto-btn" style="padding:2px 6px;font-size:10px">On</button>
      <button class="tb-btn" onclick="plClear()" style="padding:2px 6px;font-size:10px">Clear</button>
    </div>
    <div id="pl-feed" style="overflow-y:auto;flex:1"></div>
  </div>
  <div class="card" style="display:flex;flex-direction:column;overflow:hidden">
    <h3 id="pl-detail-title">Select a tick to inspect</h3>
    <div id="pl-detail" style="overflow-y:auto;flex:1;font-size:12px"></div>
  </div>
</div></div>
```

- [ ] **Step 3: Add CSS**

Before the closing `</style>` tag, add:

```css
/* Pipeline Inspector */
.pl-tick-row { display:flex;align-items:center;gap:6px;padding:4px 6px;border-bottom:1px solid #0d1117;cursor:pointer;font-size:11px; }
.pl-tick-row:hover { background:#161b22; }
.pl-tick-row.selected { background:#1a2332;border-left:2px solid #58a6ff; }
.pl-tick-num { color:#484f58;width:36px;flex-shrink:0;font-size:10px; }
.pl-tick-cycle { color:#8b949e;width:28px;flex-shrink:0; }
.pl-tick-action { flex:1;color:#c9d1d9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.pl-tick-ms { color:#484f58;font-size:10px;flex-shrink:0; }
.pl-badge { font-size:9px;padding:1px 5px;border-radius:2px;flex-shrink:0; }
.pl-badge.passed  { background:#1f3a2b;color:#3fb950; }
.pl-badge.blocked { background:#2d2210;color:#f0883e; }
.pl-badge.deduped { background:#21262d;color:#8b949e; }
.pl-badge.failed  { background:#2d1f1f;color:#f85149; }
.pl-badge.manual  { background:#1a2332;color:#58a6ff; }
.pl-stage-card { background:#0d1117;border:1px solid #21262d;border-radius:3px;padding:6px 8px;margin-bottom:6px; }
.pl-stage-header { display:flex;align-items:center;gap:6px;margin-bottom:4px; }
.pl-stage-name { font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#484f58; }
.pl-stage-label { font-size:12px;font-weight:bold;color:#c9d1d9;flex:1; }
.pl-pre { background:#161b22;border:1px solid #21262d;border-radius:2px;padding:6px 8px;font-family:'Courier New',monospace;font-size:11px;color:#8b949e;white-space:pre-wrap;word-break:break-all;max-height:180px;overflow-y:auto;margin-top:4px; }
.pl-kv { display:flex;gap:8px;padding:1px 0;font-size:11px; }
.pl-kv .k { color:#484f58;min-width:90px; }
.pl-kv .v { color:#c9d1d9; }
.pl-kv .v.ok { color:#3fb950; }
.pl-kv .v.warn { color:#f0883e; }
.pl-kv .v.bad { color:#f85149; }
```

- [ ] **Step 4: Add JavaScript before the connect() call**

Find the line `connect();` near the end of the `<script>` block. Insert the following before it:

```javascript
// Pipeline Inspector
var plSnapshots = [];
var plSelectedTick = null;
var plAutoScroll = true;

function plToggleAuto() {
  plAutoScroll = !plAutoScroll;
  document.getElementById('pl-auto-btn').textContent = plAutoScroll ? 'On' : 'Off';
  document.getElementById('pl-auto-label').style.color = plAutoScroll ? '#3fb950' : '#484f58';
}

function plClear() {
  plSnapshots = [];
  plSelectedTick = null;
  document.getElementById('pl-feed').innerHTML = '';
  document.getElementById('pl-detail').innerHTML = '';
  document.getElementById('pl-detail-title').textContent = 'Select a tick to inspect';
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function plKV(k, v, cls) {
  return '<div class="pl-kv"><span class="k">' + escHtml(String(k)) + '</span>' +
         '<span class="v' + (cls ? ' '+cls : '') + '">' + escHtml(String(v!=null?v:'--')) + '</span></div>';
}

function plValidationResult(snap) {
  var vstage = (snap.stages||[]).find(function(s){return s.name==='validation';});
  return vstage ? (vstage.data.result||'passed') : 'passed';
}

function plActionLabel(snap) {
  var vstage = (snap.stages||[]).find(function(s){return s.name==='validation';});
  if (!vstage) return '?';
  var act = vstage.data.output_action || {};
  var a = act.action||'?';
  if (a==='dig'||a==='cancel_dig') return a+' @('+act.cell_x+','+act.cell_y+')';
  if (a==='place_building') return 'place '+(act.building_id||'?')+' @('+act.cell_x+','+act.cell_y+')';
  if (a==='place_perimeter') return 'perimeter ('+act.x1+','+act.y1+')-('+act.x2+','+act.y2+')';
  return a;
}

function plAddSnapshot(snap) {
  if (plSnapshots.some(function(s){return s.tick===snap.tick&&s.cycle===snap.cycle;})) return;
  plSnapshots.push(snap);
  if (plSnapshots.length > 100) plSnapshots.shift();
  plRenderFeedRow(snap);
  if (plAutoScroll) {
    plSelectTick(snap.tick);
    var feed = document.getElementById('pl-feed');
    if (feed) feed.scrollTop = feed.scrollHeight;
  }
}

function plRenderFeedRow(snap) {
  var feed = document.getElementById('pl-feed');
  if (!feed) return;
  var vresult = plValidationResult(snap);
  var row = document.createElement('div');
  row.className = 'pl-tick-row';
  row.setAttribute('data-tick', snap.tick);
  row.onclick = function() { plSelectTick(snap.tick); };
  row.innerHTML =
    '<span class="pl-tick-num">t'+escHtml(String(snap.tick))+'</span>'+
    '<span class="pl-tick-cycle">c'+escHtml(String(snap.cycle))+'</span>'+
    '<span class="pl-tick-action">'+escHtml(plActionLabel(snap))+'</span>'+
    '<span class="pl-badge '+escHtml(vresult)+'">'+escHtml(vresult)+'</span>'+
    '<span class="pl-tick-ms">'+escHtml(String(snap.elapsed_ms))+'ms</span>';
  feed.appendChild(row);
}

function plSelectTick(tick) {
  plSelectedTick = tick;
  document.querySelectorAll('.pl-tick-row').forEach(function(row) {
    row.classList.toggle('selected', parseInt(row.getAttribute('data-tick'))===tick);
  });
  var snap = plSnapshots.find(function(s){return s.tick===tick;});
  if (!snap) return;
  document.getElementById('pl-detail-title').textContent =
    'Tick '+snap.tick+' -- Cycle '+snap.cycle+' ('+snap.elapsed_ms+'ms)';
  var detail = document.getElementById('pl-detail');
  if (!detail) return;
  detail.innerHTML = '';
  (snap.stages||[]).forEach(function(stage) { detail.appendChild(plRenderStageCard(stage)); });
}

function plRenderStageCard(stage) {
  var card = document.createElement('div');
  card.className = 'pl-stage-card';
  var d = stage.data || {};
  var header = '<div class="pl-stage-header">'+
    '<span class="pl-stage-name">'+escHtml(stage.name)+'</span>'+
    '<span class="pl-stage-label">'+escHtml(stage.label)+'</span>'+
    '</div>';
  var body = '';
  if (stage.name==='state_in') {
    body = plKV('Cycle',d.cycle)+plKV('Dupes',d.dupes)+
           plKV('O2 kg',d.o2_kg)+plKV('Alerts',d.alerts,d.alerts>0?'warn':'ok')+
           plKV('Tile window',d.tile_window);
  } else if (stage.name==='prompt') {
    body = plKV('Chars',d.chars)+plKV('Tokens (est)',d.tokens_est)+
           '<div style="font-size:10px;color:#484f58;margin:4px 0 2px">Preview:</div>'+
           '<div class="pl-pre">'+escHtml(d.preview||'')+'</div>';
  } else if (stage.name==='llm_call') {
    body = plKV('Model',d.model)+plKV('Elapsed',d.elapsed_ms+'ms')+
           '<div style="font-size:10px;color:#484f58;margin:4px 0 2px">Raw response:</div>'+
           '<div class="pl-pre">'+escHtml(d.raw_response||'')+'</div>'+
           '<div style="font-size:10px;color:#484f58;margin:4px 0 2px">Extracted JSON:</div>'+
           '<div class="pl-pre">'+escHtml(JSON.stringify(d.extracted_json,null,2)||'')+'</div>';
  } else if (stage.name==='validation') {
    var cls = d.result==='passed'?'ok':(d.result==='blocked'?'warn':'');
    body = plKV('Result',d.result,cls)+(d.reason?plKV('Reason',d.reason):'')+
           '<div style="font-size:10px;color:#484f58;margin:4px 0 2px">Input:</div>'+
           '<div class="pl-pre">'+escHtml(JSON.stringify(d.input_action,null,2)||'')+'</div>'+
           '<div style="font-size:10px;color:#484f58;margin:4px 0 2px">Output:</div>'+
           '<div class="pl-pre">'+escHtml(JSON.stringify(d.output_action,null,2)||'')+'</div>';
  } else if (stage.name==='sent') {
    body = '<div class="pl-pre">'+escHtml(JSON.stringify(d.action,null,2)||'')+'</div>';
  } else if (stage.name==='ack') {
    body = plKV('Action',d.action)+plKV('Success',d.success?'yes':'no',d.success?'ok':'bad')+
           (d.error?plKV('Error',d.error,'bad'):'');
  } else {
    body = '<div class="pl-pre">'+escHtml(JSON.stringify(d,null,2))+'</div>';
  }
  card.innerHTML = header + body;
  return card;
}
```

- [ ] **Step 5: Wire pipeline into handleMessage()**

In `handleMessage()`, add:

```javascript
  } else if (msg.type === 'pipeline') {
    plAddSnapshot(msg);
  }
```

- [ ] **Step 6: Manual test — verify tab appears and renders cleanly**

```bash
python3 examples/dashboard/server.py
```

Open `http://localhost:8181`, click Pipeline tab. Verify:
- Two-panel layout renders correctly
- Auto/Clear buttons present
- No JavaScript console errors

- [ ] **Step 7: Commit**

```bash
git add examples/dashboard/index.html
git commit -m "feat: add Pipeline inspector tab to dashboard with tick feed and stage detail"
```

---

## Task 5: End-to-end verification

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/agent/ -v
```

Expected: all 35+ tests pass.

- [ ] **Step 2: Start runner and dashboard**

Terminal 1 (runner):
```bash
GOOGLE_API_KEY=<key> python3 -m src.agent.runner --host 10.0.0.10
```

Terminal 2 (dashboard):
```bash
python3 examples/dashboard/server.py
```

- [ ] **Step 3: Verify pipeline tab populates**

Open `http://localhost:8181`, click Pipeline tab. Within 3 ticks verify:
- Tick rows appear with tick num, cycle, action label, validation badge, elapsed ms
- Clicking a row shows all 6 stage cards in the right panel
- Prompt Formatted card shows actual prompt text (should show cycle, resources, dupes)
- LLM Response card shows model name, elapsed time, raw response text
- Validation card shows passed/blocked/deduped with correct reason when applicable

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: pipeline inspector complete -- visibility into every inference stage"
```

---

## Self-Review

| Spec requirement | Task |
|----------------|------|
| decide() returns (action, prompt, raw) tuple | Task 1 |
| PipelineSnapshot with add_stage/to_dict | Task 2 |
| Stages: state_in, prompt, llm_call, validation, sent, ack | Task 2 |
| Broadcast pipeline message via relay | Task 2 |
| server.py stores last 50 snapshots | Task 3 |
| Send last 10 to new browser clients | Task 3 |
| Pipeline tab: tick feed + stage detail | Task 4 |
| Auto-scroll toggle + Clear | Task 4 |
| Validation badges: passed/blocked/deduped/failed | Task 4 |
| Prompt + response in scrollable pre blocks | Task 4 |
| Token estimation | Task 2 (tokens_est = chars//4) |
| escHtml() for all interpolated values | Task 4 |
