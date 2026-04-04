# Dashboard Speed Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add game speed controls (⏸/1x/2x/4x) and a Settings (pause) button to the far right of the existing dashboard toolbar, all mouse-clickable, no keyboard required.

**Architecture:** Dashboard sends `{"type":"action","action":"set_speed","speed":N}` via WebSocket to runner, which forwards to the game bridge. The C# mod calls `SpeedControlScreen` to change game speed. Existing toolbar buttons are unchanged.

**Tech Stack:** HTML/CSS/JS (dashboard), Python (runner + protocol), C# / Newtonsoft.Json (mod)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `examples/dashboard/index.html` | Modify | Add speed group CSS, speed buttons HTML to `#toolbar`, `setSpeed()` JS function |
| `src/agent/protocol.py` | Modify | Add `"set_speed"` to `VALID_ACTIONS` |
| `mod/ONIBridge/src/ActionCommand.cs` | Modify | Add `Speed` nullable int property |
| `mod/ONIBridge/src/ActionExecutor.cs` | Modify | Add `"set_speed"` case calling `SpeedControlScreen` |
| `tests/agent/test_protocol.py` | Modify | Add test for `set_speed` action validation |

---

### Task 1: Add `set_speed` to protocol

**Files:**
- Modify: `src/agent/protocol.py`
- Test: `tests/agent/test_protocol.py`

- [ ] **Step 1: Write failing test**

Open `tests/agent/test_protocol.py` and add:

```python
def test_build_set_speed_action():
    action = build_action("set_speed", speed=2)
    assert action["action"] == "set_speed"
    assert action["speed"] == 2


def test_set_speed_is_valid_action():
    from src.agent.protocol import VALID_ACTIONS
    assert "set_speed" in VALID_ACTIONS


def test_build_unknown_action_raises():
    with pytest.raises(ValueError):
        build_action("invalid_action")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/agent/test_protocol.py::test_build_set_speed_action tests/agent/test_protocol.py::test_set_speed_is_valid_action -v
```

Expected: FAIL — `"set_speed"` not in `VALID_ACTIONS`.

- [ ] **Step 3: Add `set_speed` to `VALID_ACTIONS`**

In `src/agent/protocol.py`, change:

```python
VALID_ACTIONS = {"place_building", "dig", "cancel_dig", "set_priority", "no_op"}
```

to:

```python
VALID_ACTIONS = {"place_building", "dig", "cancel_dig", "set_priority", "no_op", "set_speed"}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/agent/test_protocol.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent/protocol.py tests/agent/test_protocol.py
git commit -m "feat: add set_speed to protocol VALID_ACTIONS"
```

---

### Task 2: Add `Speed` field to C# ActionCommand

**Files:**
- Modify: `mod/ONIBridge/src/ActionCommand.cs`

- [ ] **Step 1: Add `Speed` property**

Open `mod/ONIBridge/src/ActionCommand.cs`. After the `Priority` property, add:

```csharp
// Game speed: 0=paused, 1=normal, 2=fast, 3=ultra
[JsonProperty("speed")]
public int? Speed { get; set; }
```

The full file should now look like:

```csharp
using Newtonsoft.Json;

namespace ONIBridge
{
    public class ActionCommand
    {
        [JsonProperty("type")]
        public string Type { get; set; } = "action";

        [JsonProperty("action")]
        public string Action { get; set; } = "";

        [JsonProperty("building_id")]
        public string? BuildingId { get; set; }

        [JsonProperty("cell_x")]
        public int CellX { get; set; }

        [JsonProperty("cell_y")]
        public int CellY { get; set; }

        [JsonProperty("priority")]
        public int Priority { get; set; } = 5;

        [JsonProperty("duplicant_id")]
        public int DuplicantId { get; set; } = -1;

        [JsonProperty("skill")]
        public string? Skill { get; set; }

        // Game speed: 0=paused, 1=normal, 2=fast, 3=ultra
        [JsonProperty("speed")]
        public int? Speed { get; set; }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add mod/ONIBridge/src/ActionCommand.cs
git commit -m "feat: add Speed field to ActionCommand for set_speed action"
```

---

### Task 3: Add `set_speed` case to C# ActionExecutor

**Files:**
- Modify: `mod/ONIBridge/src/ActionExecutor.cs`

- [ ] **Step 1: Add `set_speed` case to `Execute()`**

In `mod/ONIBridge/src/ActionExecutor.cs`, inside the `switch (cmd.Action)` block, add a new case after `"no_op"` and before `default`:

```csharp
case "set_speed":
{
    int speed = cmd.Speed ?? 1;
    speed = System.Math.Max(0, System.Math.Min(3, speed));
    try
    {
        if (speed == 0)
        {
            // Pause
            if (SpeedControlScreen.Instance != null)
                SpeedControlScreen.Instance.Pause();
        }
        else
        {
            // Speed 1=normal, 2=fast, 3=ultra — matches SpeedControlScreen button indices
            if (SpeedControlScreen.Instance != null)
                SpeedControlScreen.Instance.SetSpeed(speed);
        }
        BridgeServer.Instance.SendAck(cmd.Action, true);
        Debug.Log($"[ONIBridge] Speed set to {speed}");
    }
    catch (System.Exception ex)
    {
        Debug.LogWarning($"[ONIBridge] set_speed failed: {ex.Message}");
        BridgeServer.Instance.SendAck(cmd.Action, false, ex.Message);
    }
    break;
}
```

- [ ] **Step 2: Rebuild the mod**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge
dotnet build
```

Expected: Build succeeded, 0 errors.

- [ ] **Step 3: Deploy and verify**

Copy the rebuilt DLL to the ONI mods folder on the Windows PC. Load the game. In the runner logs, manually send a `set_speed` action and confirm the game speed changes without any `[ONIBridge]` error in the game log.

- [ ] **Step 4: Commit**

```bash
git add mod/ONIBridge/src/ActionExecutor.cs
git commit -m "feat: handle set_speed action in ActionExecutor via SpeedControlScreen"
```

---

### Task 4: Add speed controls to dashboard toolbar

**Files:**
- Modify: `examples/dashboard/index.html`

- [ ] **Step 1: Add CSS for speed controls**

In `examples/dashboard/index.html`, inside the `<style>` block, after the `.tb-sep` rule (around line 186), add:

```css
/* ── Speed controls (right side of toolbar) ── */
.speed-group {
  display: flex;
  border: 1px solid #30363d;
  border-radius: 4px;
  overflow: hidden;
  flex-shrink: 0;
  margin: 0 4px;
}
.tb-speed-btn {
  background: #21262d;
  color: #8b949e;
  border: none;
  border-right: 1px solid #30363d;
  padding: 3px 9px;
  cursor: pointer;
  font-family: inherit;
  font-size: 11px;
  transition: background 0.1s;
}
.tb-speed-btn:last-child { border-right: none; }
.tb-speed-btn:hover { background: #30363d; color: #c9d1d9; }
.tb-speed-btn.active { background: #1f3a2b; color: #3fb950; font-weight: bold; }
.tb-speed-btn.paused { background: #2d1f1f; color: #f85149; }
#tb-speed-status {
  font-size: 10px;
  color: #3fb950;
  white-space: nowrap;
  flex-shrink: 0;
  margin-right: 4px;
}
#tb-speed-status.paused { color: #f85149; }
```

- [ ] **Step 2: Add speed controls HTML to `#toolbar`**

In `examples/dashboard/index.html`, find the `#toolbar` div (around line 289). It currently ends with:

```html
  <button class="tb-action" id="btn-noop"><span class="tb-icon">⏸️</span><span class="tb-label">No-op</span></button>
</div>
```

Change it to:

```html
  <button class="tb-action" id="btn-noop"><span class="tb-icon">⏸️</span><span class="tb-label">No-op</span></button>

  <!-- Spacer pushes speed controls to far right -->
  <div style="flex:1"></div>

  <!-- Speed status label -->
  <span id="tb-speed-status">● 1x</span>

  <!-- Speed segmented control -->
  <div class="speed-group">
    <button class="tb-speed-btn" id="spd-0" onclick="setSpeed(0)" title="Pause">⏸</button>
    <button class="tb-speed-btn active" id="spd-1" onclick="setSpeed(1)" title="Normal (1x)">1x</button>
    <button class="tb-speed-btn" id="spd-2" onclick="setSpeed(2)" title="Fast (2x)">2x</button>
    <button class="tb-speed-btn" id="spd-3" onclick="setSpeed(3)" title="Ultra (4x)">4x</button>
  </div>

  <div class="tb-sep"></div>

  <!-- Settings = pause -->
  <button class="tb-btn" onclick="setSpeed(0)" title="Pause game">
    <span class="tb-icon">⚙️</span>
    <span class="tb-label">Settings</span>
  </button>
</div>
```

- [ ] **Step 3: Add `setSpeed()` JavaScript function**

In `examples/dashboard/index.html`, inside the `<script>` block, after the `runnerStop()` function (around line 611), add:

```javascript
// ── Speed controls ────────────────────────────────────────────────────────
var currentSpeed = 1;

function setSpeed(speed) {
  currentSpeed = speed;
  send({ type: "action", action: "set_speed", speed: speed });
  renderSpeedButtons(speed);
}

function renderSpeedButtons(speed) {
  [0, 1, 2, 3].forEach(function(s) {
    var btn = document.getElementById("spd-" + s);
    if (!btn) return;
    btn.classList.remove("active", "paused");
    if (s === speed) {
      btn.classList.add(speed === 0 ? "paused" : "active");
    }
  });
  var status = document.getElementById("tb-speed-status");
  if (!status) return;
  if (speed === 0) {
    status.textContent = "● Paused";
    status.classList.add("paused");
  } else {
    var labels = ["", "1x", "2x", "4x"];
    status.textContent = "● " + (labels[speed] || speed + "x");
    status.classList.remove("paused");
  }
}
```

- [ ] **Step 4: Verify in browser**

Open http://localhost:8181 in the browser.

- Bottom toolbar should show existing game buttons on the left
- Far right shows: `● 1x` label, then `⏸ | 1x | 2x | 4x` segmented control, separator, `⚙️ Settings` button
- Click `⏸` — button turns red, label shows `● Paused`
- Click `1x` — button turns green, label shows `● 1x`
- Click `⚙️ Settings` — same effect as `⏸` (pauses game)
- Browser console should show no JS errors

- [ ] **Step 5: Commit**

```bash
git add examples/dashboard/index.html
git commit -m "feat: add speed controls and settings button to dashboard toolbar"
```

---

### Task 5: End-to-end verification

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 2: Live test with game running**

1. Start the dashboard: `GOOGLE_API_KEY=<key> python3 examples/dashboard/server.py`
2. Click Start in the dashboard
3. Wait for the runner to connect and start ticking
4. Click `⏸` in the toolbar — game should pause (duplicants stop moving)
5. Click `1x` — game resumes at normal speed
6. Click `4x` — game speeds up to ultra
7. Click `⚙️ Settings` — game pauses

Check runner logs for `set_speed` actions being forwarded. No errors should appear.

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "feat: dashboard speed controls — end-to-end verified"
```
