# Dashboard Speed Controls — Design Spec

**Date:** 2026-04-04
**Status:** Approved

---

## Overview

Add game speed controls and a Settings (pause) button to the far right of the existing dashboard toolbar. All controls are mouse-clickable. No keyboard required. The existing game action buttons (Base, O2, Power, Dig, etc.) are unchanged.

---

## UI Changes (index.html)

### Toolbar Addition

Inside `#toolbar`, after the existing buttons, add:

```html
<!-- spacer pushes speed controls to far right -->
<div style="flex:1"></div>

<!-- speed status indicator -->
<span id="tb-speed-status" class="tb-speed-status">● 1x</span>

<!-- speed segmented control -->
<div class="speed-group">
  <button class="tb-speed-btn" id="spd-0" onclick="setSpeed(0)" title="Pause">⏸</button>
  <button class="tb-speed-btn active" id="spd-1" onclick="setSpeed(1)" title="Normal (1x)">1x</button>
  <button class="tb-speed-btn" id="spd-2" onclick="setSpeed(2)" title="Fast (2x)">2x</button>
  <button class="tb-speed-btn" id="spd-3" onclick="setSpeed(3)" title="Ultra (4x)">4x</button>
</div>

<!-- separator -->
<div class="tb-sep"></div>

<!-- game settings = pause -->
<button class="tb-btn" onclick="setSpeed(0)" title="Pause game (opens settings on game side)">
  <span class="tb-icon">⚙️</span>
  <span class="tb-label">Settings</span>
</button>
```

### New CSS Classes

```css
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
.tb-speed-status {
  font-size: 10px;
  color: #3fb950;
  white-space: nowrap;
  flex-shrink: 0;
  margin-right: 4px;
}
.tb-speed-status.paused { color: #f85149; }
```

### JavaScript

```javascript
var currentSpeed = 1;

function setSpeed(speed) {
  currentSpeed = speed;
  send({ type: "action", action: "set_speed", speed: speed });
  renderSpeedButtons(speed);
}

function renderSpeedButtons(speed) {
  [0, 1, 2, 3].forEach(function(s) {
    var btn = document.getElementById("spd-" + s);
    btn.classList.remove("active", "paused");
    if (s === speed) {
      btn.classList.add(speed === 0 ? "paused" : "active");
    }
  });
  var status = document.getElementById("tb-speed-status");
  if (speed === 0) {
    status.textContent = "● Paused";
    status.classList.add("paused");
  } else {
    var labels = ["", "1x", "2x", "4x"];
    status.textContent = "● " + labels[speed];
    status.classList.remove("paused");
  }
}
```

---

## Protocol Change

New action type added to `VALID_ACTIONS` in `src/agent/protocol.py`:

```python
VALID_ACTIONS = {"place_building", "dig", "cancel_dig", "set_priority", "no_op", "set_speed"}
```

Action shape:
```json
{"type": "action", "action": "set_speed", "speed": 0}
```

`speed` values: `0` = paused, `1` = normal, `2` = fast, `3` = ultra.

---

## Runner Changes (runner.py)

`set_speed` is forwarded to the game bridge like any other action. No special handling needed in the runner — `client.send_action(action)` already sends arbitrary action dicts.

The dedup logic already skips `no_op`; `set_speed` is also excluded from AI decision flow (manual-only). The runner only emits AI actions; speed control is always a manual dashboard action.

---

## C# Mod Changes (ActionExecutor.cs)

New case in `ActionExecutor.Execute()`:

```csharp
case "set_speed":
{
    int speed = cmd.Speed ?? 1;
    // SpeedControlScreen maps: 0=paused, 1=normal, 2=fast, 3=ultra
    if (speed == 0)
        SpeedControlScreen.Instance?.Pause();
    else
        SpeedControlScreen.Instance?.SetSpeed(speed);
    break;
}
```

`ActionCommand.cs` gains an optional `Speed` field:

```csharp
[JsonProperty("speed")]
public int? Speed { get; set; }
```

---

## Files Changed

| File | Change |
|------|--------|
| `examples/dashboard/index.html` | Add speed group + Settings button to right of `#toolbar`, add CSS + JS |
| `src/agent/protocol.py` | Add `"set_speed"` to `VALID_ACTIONS` |
| `mod/ONIBridge/src/ActionCommand.cs` | Add `Speed` property |
| `mod/ONIBridge/src/ActionExecutor.cs` | Add `set_speed` case |

---

## Out of Scope

- Full in-game settings panel interaction (too complex, low ROI for now)
- Speed state sync from game to dashboard (game doesn't currently report current speed in state)
- Keyboard shortcuts (user wants mouse-only interaction)
