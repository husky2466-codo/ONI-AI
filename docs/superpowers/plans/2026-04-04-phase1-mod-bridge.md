# ONI Bridge Phase 1 — Mod Bridge + Stub Agent

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the C# Harmony mod building, loading in ONI on Ubuntu, serializing real game state, executing the five action types, and verified end-to-end with a Python stub client that connects, receives state, sends a `no_op`, and gets an ack back.

**Architecture:** A Harmony mod opens a TCP server on port 9999 inside the ONI game process. A Python client on DGX Spark 1 connects over the LAN, receives newline-delimited JSON state snapshots every 10 game ticks, and sends back JSON action commands that the mod executes on the Unity main thread.

**Tech Stack:** C# net471 / Mono (mod), dotnet CLI + Rider (build), Python 3.11 + asyncio (stub client), ONI Harmony 2.0 (built into game), Newtonsoft.Json (built into game)

---

## File Map

### C# Mod — `mod/ONIBridge/`
| File | Responsibility |
|---|---|
| `ONIBridge.csproj` | Build config — already scaffolded, will be modified |
| `mod_info.yaml` | ONI mod loader metadata — already scaffolded |
| `mod.yaml` | Display metadata — already scaffolded |
| `lib/` | Game DLLs copied from Ubuntu ONI install — must populate |
| `src/ONIBridgeMod.cs` | `UserMod2` entry point — already scaffolded, no changes needed |
| `src/BridgeServer.cs` | TCP server + action queue — already scaffolded, complete |
| `src/GameTickPatch.cs` | Harmony `Game.Update` postfix — already scaffolded, complete |
| `src/ActionCommand.cs` | Command schema — already scaffolded, complete |
| `src/ActionExecutor.cs` | Stub → real ONI API calls — **needs implementation** |
| `src/StateSerializer.cs` | Stub → real ONI API calls — **needs implementation** |
| `src/AckMessage.cs` | New: ACK response schema |

### Python Stub Client — `src/agent/`
| File | Responsibility |
|---|---|
| `src/agent/__init__.py` | Package marker |
| `src/agent/client.py` | TCP client — connect, receive state, send actions |
| `src/agent/protocol.py` | Parse/validate state JSON, build action JSON |
| `tests/agent/test_protocol.py` | Unit tests for protocol parsing |

---

## Task 1: Copy Game DLLs to lib/

The `.csproj` references game DLLs from `mod/ONIBridge/lib/`. These must be copied from the Ubuntu ONI install before the project will build.

**Files:**
- Populate: `mod/ONIBridge/lib/` (6 DLL files)

- [ ] **Step 1: Run copy script from Ubuntu**

SSH to Ubuntu and run:
```bash
ssh myroproductions@<ubuntu-ip>
cd /tmp
# Find the Managed folder
ls ~/.steam/steam/steamapps/common/OxygenNotIncluded/OxygenNotIncluded_Data/Managed/ | grep -E "Assembly-CSharp|0Harmony|UnityEngine.dll|UnityEngine.CoreModule|Newtonsoft"
```
Expected output includes: `Assembly-CSharp.dll`, `Assembly-CSharp-firstpass.dll`, `0Harmony.dll`, `UnityEngine.dll`, `UnityEngine.CoreModule.dll`, `Newtonsoft.Json.dll`

- [ ] **Step 2: SCP DLLs to Mac**

From Mac terminal:
```bash
UBUNTU_IP=<ubuntu-ip>
MANAGED="$HOME/.steam/steam/steamapps/common/OxygenNotIncluded/OxygenNotIncluded_Data/Managed"
LIB="/Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge/lib"

scp myroproductions@$UBUNTU_IP:"$MANAGED/Assembly-CSharp.dll \
  $MANAGED/Assembly-CSharp-firstpass.dll \
  $MANAGED/0Harmony.dll \
  $MANAGED/UnityEngine.dll \
  $MANAGED/UnityEngine.CoreModule.dll \
  $MANAGED/Newtonsoft.Json.dll" "$LIB/"
```

- [ ] **Step 3: Verify all 6 DLLs are present**

```bash
ls -lh /Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge/lib/
```
Expected: 6 files, all non-zero size. `Assembly-CSharp.dll` will be the largest (several MB).

- [ ] **Step 4: Commit**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
# Add lib/ DLLs — these are gitignored by default, add explicitly
echo "mod/ONIBridge/lib/*.dll" >> .gitignore  # keep gitignored — large binaries
git add .gitignore
git commit -m "chore: gitignore game DLLs in mod/lib"
```

---

## Task 2: First Build — Verify Project Compiles

- [ ] **Step 1: Restore NuGet packages**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge
dotnet restore
```
Expected: `Restore complete.` — downloads `Microsoft.NETFramework.ReferenceAssemblies.net471` and `Lib.Harmony.Ref`.

- [ ] **Step 2: Build**

```bash
dotnet build -c Debug
```
Expected: `Build succeeded. 0 Error(s)` — warnings about nullable refs are fine. Any error about missing DLLs means Task 1 didn't complete.

- [ ] **Step 3: Open in Rider on Mac**

Open Rider → File → Open → select `/Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge/ONIBridge.csproj`

In Rider: right-click any ONI type (e.g. `Game` in `GameTickPatch.cs`) → Go To → Declaration. Rider will decompile `Assembly-CSharp.dll` and show the source. You'll use this to find the real API signatures for Tasks 4 and 5.

- [ ] **Step 4: Commit**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
git add mod/ONIBridge/
git commit -m "build: ONIBridge mod compiles against net471 game DLLs"
```

---

## Task 3: Add ACK Message Type

The current scaffold sends state but has no ACK schema. Add it before implementing ActionExecutor.

**Files:**
- Create: `mod/ONIBridge/src/AckMessage.cs`
- Modify: `mod/ONIBridge/src/BridgeServer.cs` — add `SendAck()` method

- [ ] **Step 1: Create AckMessage.cs**

```csharp
// mod/ONIBridge/src/AckMessage.cs
using Newtonsoft.Json;

namespace ONIBridge
{
    public class AckMessage
    {
        [JsonProperty("type")]
        public string Type { get; } = "ack";

        [JsonProperty("action")]
        public string Action { get; set; } = "";

        [JsonProperty("success")]
        public bool Success { get; set; }

        [JsonProperty("error")]
        public string? Error { get; set; }
    }
}
```

- [ ] **Step 2: Add SendAck to BridgeServer.cs**

In `BridgeServer.cs`, after the `SendState` method, add:

```csharp
public void SendAck(string action, bool success, string? error = null)
{
    if (_connectedClient?.Connected != true) return;
    try
    {
        var msg = JsonConvert.SerializeObject(new AckMessage
        {
            Action = action,
            Success = success,
            Error = error
        });
        var bytes = System.Text.Encoding.UTF8.GetBytes(msg + "\n");
        _connectedClient.GetStream().Write(bytes, 0, bytes.Length);
    }
    catch (System.Exception ex)
    {
        Debug.LogWarning($"[ONIBridge] SendAck failed: {ex.Message}");
    }
}
```

- [ ] **Step 3: Build to verify**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge
dotnet build -c Debug
```
Expected: `Build succeeded. 0 Error(s)`

- [ ] **Step 4: Commit**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
git add mod/ONIBridge/src/AckMessage.cs mod/ONIBridge/src/BridgeServer.cs
git commit -m "feat: add ACK message type and SendAck to BridgeServer"
```

---

## Task 4: Implement StateSerializer with Real Game APIs

Replace all stubs in `StateSerializer.cs` with calls to ONI's actual C# APIs.

**How to find API signatures:** In Rider, Ctrl+Click any game type to decompile it. Key types to explore:
- `GameClock` — cycle and time
- `WorldInventory` — resource amounts
- `Components.MinionIdentities` — duplicant list
- `Components.BuildingCompletes` — building list
- `ColonyDiagnosticUtility` — alert strings
- `Grid` — cell data, `Grid.XYToCell(x, y)`

**Files:**
- Modify: `mod/ONIBridge/src/StateSerializer.cs`

- [ ] **Step 1: Replace GetCycle() and GetTime()**

```csharp
private static int GetCycle()
{
    return GameClock.Instance != null ? (int)GameClock.Instance.GetCycle() : 0;
}

private static float GetTime()
{
    return GameClock.Instance != null ? GameClock.Instance.GetTime() : 0f;
}
```

- [ ] **Step 2: Replace GetResources()**

```csharp
private static object GetResources()
{
    var inv = WorldInventory.Instance;
    if (inv == null) return new { oxygen_kg = 0f, water_kg = 0f, food_kcal = 0f, power_kw = 0f, co2_kg = 0f };

    float OxygenTag = inv.GetAmount(SimHashes.Oxygen.CreateTag(), out _);
    float WaterTag = inv.GetAmount(SimHashes.Water.CreateTag(), out _);
    float Co2Tag = inv.GetAmount(SimHashes.CarbonDioxide.CreateTag(), out _);

    // Food is tracked in kcal via RationTracker
    float food = RationTracker.Get() != null ? RationTracker.Get().GetTotalRations() : 0f;

    // Power: sum operational generators minus consumers — use CircuitManager
    float power = 0f;
    var circuits = Game.Instance?.circuitManager;
    if (circuits != null)
    {
        for (int i = 0; i < circuits.circuitCount; i++)
        {
            var circuit = circuits.GetCircuit(i);
            if (circuit != null) power += circuit.wattsGeneratedByGenerators - circuit.wattsUsedByConsumers;
        }
    }

    return new
    {
        oxygen_kg = OxygenTag / 1000f,
        water_kg = WaterTag / 1000f,
        food_kcal = food,
        power_kw = power / 1000f,
        co2_kg = Co2Tag / 1000f,
    };
}
```

- [ ] **Step 3: Replace GetDuplicants()**

```csharp
private static System.Collections.Generic.List<object> GetDuplicants()
{
    var result = new System.Collections.Generic.List<object>();
    if (Components.MinionIdentities == null) return result;

    foreach (MinionIdentity minion in Components.MinionIdentities)
    {
        if (minion == null) continue;
        var pos = minion.transform.position;
        float stress = 0f;
        float health = 0f;
        string currentTask = "idle";

        var stressMon = minion.GetSMI<StressMonitor.Instance>();
        if (stressMon != null) stress = stressMon.GetStress();

        var hp = minion.GetComponent<Health>();
        if (hp != null) health = hp.hitPoints;

        var chore = minion.GetComponent<ChoreDriver>()?.GetCurrentChore();
        if (chore != null) currentTask = chore.choreType?.Id ?? "unknown";

        result.Add(new
        {
            id = minion.GetInstanceID(),
            name = minion.name,
            x = (int)pos.x,
            y = (int)pos.y,
            stress = System.Math.Round(stress, 2),
            health = System.Math.Round(health, 1),
            current_task = currentTask,
        });
    }
    return result;
}
```

- [ ] **Step 4: Replace GetAlerts()**

```csharp
private static System.Collections.Generic.List<string> GetAlerts()
{
    var alerts = new System.Collections.Generic.List<string>();
    if (ColonyDiagnosticUtility.Instance == null) return alerts;

    var worldId = ClusterManager.Instance?.activeWorldId ?? 0;
    var diagnostics = ColonyDiagnosticUtility.Instance.GetDiagnostics(worldId);
    if (diagnostics == null) return alerts;

    foreach (var diag in diagnostics)
    {
        if (diag?.LatestResult == null) continue;
        if (diag.LatestResult.opinion == ColonyDiagnostic.DiagnosticResult.Opinion.Bad ||
            diag.LatestResult.opinion == ColonyDiagnostic.DiagnosticResult.Opinion.DangerouslyBad)
        {
            alerts.Add($"{diag.name}: {diag.LatestResult.Message}");
        }
    }
    return alerts;
}
```

- [ ] **Step 5: Replace GetBuildings() — add this method to StateSerializer**

```csharp
private static System.Collections.Generic.List<object> GetBuildings()
{
    var result = new System.Collections.Generic.List<object>();
    if (Components.BuildingCompletes == null) return result;

    foreach (BuildingComplete b in Components.BuildingCompletes)
    {
        if (b == null) continue;
        var pos = b.transform.position;
        var op = b.GetComponent<Operational>();
        result.Add(new
        {
            type = b.Def?.PrefabID ?? "unknown",
            x = (int)pos.x,
            y = (int)pos.y,
            operational = op != null && op.IsOperational,
        });
    }
    return result;
}
```

- [ ] **Step 6: Update Serialize() to include buildings**

Replace the `Serialize()` method body:

```csharp
public static object Serialize()
{
    return new
    {
        cycle = GetCycle(),
        time = GetTime(),
        resources = GetResources(),
        duplicants = GetDuplicants(),
        buildings = GetBuildings(),
        alerts = GetAlerts(),
    };
}
```

- [ ] **Step 7: Build**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge
dotnet build -c Debug
```
Expected: `Build succeeded.` — if any method doesn't exist, use Rider to decompile the correct type and find the actual signature. The `StressMonitor.Instance` pattern in particular may differ; check via Ctrl+Click on `StressMonitor` in Rider.

- [ ] **Step 8: Commit**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
git add mod/ONIBridge/src/StateSerializer.cs
git commit -m "feat: implement StateSerializer with real ONI game APIs"
```

---

## Task 5: Implement ActionExecutor with Real ONI APIs

Replace stubs in `ActionExecutor.cs` with calls to ONI's actual build/dig/priority systems. Also wire up ACK sending.

**Files:**
- Modify: `mod/ONIBridge/src/ActionExecutor.cs`

- [ ] **Step 1: Replace PlaceBuilding()**

```csharp
private static void PlaceBuilding(ActionCommand cmd)
{
    if (string.IsNullOrEmpty(cmd.BuildingId))
    {
        BridgeServer.Instance.SendAck(cmd.Action, false, "building_id is required");
        return;
    }

    var def = Assets.GetBuildingDef(cmd.BuildingId);
    if (def == null)
    {
        BridgeServer.Instance.SendAck(cmd.Action, false, $"Unknown building: {cmd.BuildingId}");
        return;
    }

    int cell = Grid.XYToCell(cmd.CellX, cmd.CellY);
    if (!Grid.IsValidCell(cell))
    {
        BridgeServer.Instance.SendAck(cmd.Action, false, $"Invalid cell ({cmd.CellX},{cmd.CellY})");
        return;
    }

    // Check buildable — same check the UI does
    if (!def.IsValidPlaceLocation(null, cell, Orientation.Neutral, out string reason))
    {
        BridgeServer.Instance.SendAck(cmd.Action, false, $"Cannot place: {reason}");
        return;
    }

    // Use the same code path as the build tool
    var selected_elements = def.DefaultElements();
    def.TryPlace(null, cell, Orientation.Neutral, selected_elements, 0);
    BridgeServer.Instance.SendAck(cmd.Action, true);
    Debug.Log($"[ONIBridge] Placed {cmd.BuildingId} at ({cmd.CellX},{cmd.CellY})");
}
```

- [ ] **Step 2: Replace Dig()**

```csharp
private static void Dig(ActionCommand cmd)
{
    int cell = Grid.XYToCell(cmd.CellX, cmd.CellY);
    if (!Grid.IsValidCell(cell))
    {
        BridgeServer.Instance.SendAck(cmd.Action, false, $"Invalid cell ({cmd.CellX},{cmd.CellY})");
        return;
    }

    if (!Grid.IsSolid(cell))
    {
        BridgeServer.Instance.SendAck(cmd.Action, false, "Cell is not solid — nothing to dig");
        return;
    }

    // Mark cell for digging — same as the dig tool
    Grid.Objects[cell, (int)ObjectLayer.DigPlacer] = null;
    Diggable diggable = Grid.Objects[cell, (int)ObjectLayer.Pickupables] as Diggable;
    if (diggable == null)
    {
        diggable = GameUtil.KInstantiate(
            Assets.GetPrefab(new Tag("DigPlacer")),
            Grid.CellToPosCBC(cell, Grid.SceneLayer.Building),
            Grid.SceneLayer.Building
        ).GetComponent<Diggable>();
        diggable.gameObject.SetActive(true);
    }
    BridgeServer.Instance.SendAck(cmd.Action, true);
    Debug.Log($"[ONIBridge] Dig queued at ({cmd.CellX},{cmd.CellY})");
}
```

- [ ] **Step 3: Replace CancelDig()**

```csharp
private static void CancelDig(ActionCommand cmd)
{
    int cell = Grid.XYToCell(cmd.CellX, cmd.CellY);
    if (!Grid.IsValidCell(cell))
    {
        BridgeServer.Instance.SendAck(cmd.Action, false, $"Invalid cell ({cmd.CellX},{cmd.CellY})");
        return;
    }

    var digPlacer = Grid.Objects[cell, (int)ObjectLayer.DigPlacer];
    if (digPlacer != null)
    {
        UnityEngine.Object.Destroy(digPlacer);
        BridgeServer.Instance.SendAck(cmd.Action, true);
    }
    else
    {
        BridgeServer.Instance.SendAck(cmd.Action, false, "No dig order at cell");
    }
}
```

- [ ] **Step 4: Replace SetPriority()**

```csharp
private static void SetPriority(ActionCommand cmd)
{
    int cell = Grid.XYToCell(cmd.CellX, cmd.CellY);
    if (!Grid.IsValidCell(cell))
    {
        BridgeServer.Instance.SendAck(cmd.Action, false, $"Invalid cell ({cmd.CellX},{cmd.CellY})");
        return;
    }

    int priority = System.Math.Clamp(cmd.Priority, 1, 9);

    // Try to find a prioritizable object at this cell
    var go = Grid.Objects[cell, (int)ObjectLayer.Building]
          ?? Grid.Objects[cell, (int)ObjectLayer.DigPlacer]
          ?? Grid.Objects[cell, (int)ObjectLayer.PlacedObject];

    if (go == null)
    {
        BridgeServer.Instance.SendAck(cmd.Action, false, "No prioritizable object at cell");
        return;
    }

    var prioritizable = go.GetComponent<Prioritizable>();
    if (prioritizable == null)
    {
        BridgeServer.Instance.SendAck(cmd.Action, false, "Object is not prioritizable");
        return;
    }

    prioritizable.SetMasterPriority(new PrioritySetting(PriorityScreen.PriorityClass.basic, priority));
    BridgeServer.Instance.SendAck(cmd.Action, true);
    Debug.Log($"[ONIBridge] Priority set to {priority} at ({cmd.CellX},{cmd.CellY})");
}
```

- [ ] **Step 5: Update Execute() to send ACK for no_op**

In the `Execute()` switch, update the `no_op` case:
```csharp
case "no_op":
    BridgeServer.Instance.SendAck(cmd.Action, true);
    break;
```

- [ ] **Step 6: Build**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge
dotnet build -c Debug
```
Expected: `Build succeeded.` — if `System.Math.Clamp` isn't available in net471 (it's .NET Core), replace with `Math.Max(1, Math.Min(9, cmd.Priority))`.

- [ ] **Step 7: Commit**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
git add mod/ONIBridge/src/ActionExecutor.cs
git commit -m "feat: implement ActionExecutor with real ONI build/dig/priority APIs"
```

---

## Task 6: Deploy and Smoke-Test Mod in ONI on Ubuntu

- [ ] **Step 1: Build release DLL on Mac**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge
dotnet build -c Release -o /tmp/ONIBridge-build
ls /tmp/ONIBridge-build/ONIBridge.dll
```

- [ ] **Step 2: SCP mod to Ubuntu mod dev folder**

```bash
UBUNTU_IP=<ubuntu-ip>
MOD_DIR="/home/myroproductions/.config/unity3d/Klei/Oxygen Not Included/mods/dev/ONIBridge"

ssh myroproductions@$UBUNTU_IP "mkdir -p '$MOD_DIR'"
scp /tmp/ONIBridge-build/ONIBridge.dll myroproductions@$UBUNTU_IP:"'$MOD_DIR/'"
scp /Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge/mod_info.yaml myroproductions@$UBUNTU_IP:"'$MOD_DIR/'"
scp /Volumes/DevDrive-M4Pro/Projects/ONI-AI/mod/ONIBridge/mod.yaml myroproductions@$UBUNTU_IP:"'$MOD_DIR/'"
```

- [ ] **Step 3: Tail the log before launching ONI**

On Ubuntu (in a terminal or via SSH):
```bash
tail -f "/home/myroproductions/.config/unity3d/Klei/Oxygen Not Included/Player.log"
```

- [ ] **Step 4: Launch ONI on Ubuntu and load a save**

Open ONI from Steam on the Ubuntu desktop. Load any existing save or start a new game.

- [ ] **Step 5: Verify mod loaded in log**

Watch the tailed log for:
```
[ONIBridge] Mod loaded — starting bridge server...
[ONIBridge] Listening on port 9999
```
If you see `[ONIBridge] Mod loaded` but an exception after it, read the stack trace — most likely a missing method in `StateSerializer` or `ActionExecutor` that needs the real API signature found via Rider decompile.

- [ ] **Step 6: Verify port is open**

From Mac or DGX, test TCP connectivity:
```bash
nc -zv <ubuntu-ip> 9999
```
Expected: `Connection to <ubuntu-ip> 9999 port [tcp/*] succeeded!`

---

## Task 7: Python Stub Client with Protocol Tests

Build the Python TCP client with tests before wiring it to the actual game.

**Files:**
- Create: `src/agent/__init__.py`
- Create: `src/agent/protocol.py`
- Create: `src/agent/client.py`
- Create: `tests/agent/__init__.py`
- Create: `tests/agent/test_protocol.py`

- [ ] **Step 1: Write failing tests for protocol.py**

```python
# tests/agent/test_protocol.py
import json
import pytest
from src.agent.protocol import (
    parse_state_message,
    build_action,
    build_no_op,
    StateMessage,
    ActionMessage,
)


def test_parse_valid_state_message():
    raw = json.dumps({
        "type": "state",
        "cycle": 3,
        "data": {
            "resources": {"oxygen_kg": 12.5, "water_kg": 80.0, "food_kcal": 2000.0, "power_kw": 1.2, "co2_kg": 0.3},
            "duplicants": [{"id": 1, "name": "Higby", "x": 10, "y": 8, "stress": 0.1, "health": 100.0, "current_task": "dig"}],
            "buildings": [{"type": "OxygenDiffuser", "x": 5, "y": 5, "operational": True}],
            "alerts": [],
        }
    })
    msg = parse_state_message(raw)
    assert isinstance(msg, StateMessage)
    assert msg.cycle == 3
    assert msg.data["resources"]["oxygen_kg"] == 12.5
    assert msg.data["duplicants"][0]["name"] == "Higby"


def test_parse_ignores_non_state_type():
    raw = json.dumps({"type": "ack", "action": "no_op", "success": True})
    msg = parse_state_message(raw)
    assert msg is None


def test_parse_invalid_json_raises():
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_state_message("not json {{{")


def test_build_no_op():
    action = build_no_op()
    assert action["type"] == "action"
    assert action["action"] == "no_op"


def test_build_place_building():
    action = build_action("place_building", building_id="OxygenDiffuser", cell_x=12, cell_y=8)
    assert action["action"] == "place_building"
    assert action["building_id"] == "OxygenDiffuser"
    assert action["cell_x"] == 12
    assert action["cell_y"] == 8


def test_build_dig():
    action = build_action("dig", cell_x=5, cell_y=3)
    assert action["action"] == "dig"
    assert action["cell_x"] == 5
    assert action["cell_y"] == 3


def test_build_set_priority():
    action = build_action("set_priority", cell_x=5, cell_y=3, priority=7)
    assert action["action"] == "set_priority"
    assert action["priority"] == 7


def test_build_action_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown action"):
        build_action("fly_rocket")
```

- [ ] **Step 2: Run to verify all tests fail**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
pip install pytest --quiet
pytest tests/agent/test_protocol.py -v
```
Expected: `ModuleNotFoundError` or `ImportError` — protocol.py doesn't exist yet.

- [ ] **Step 3: Create src/agent/__init__.py and tests/agent/__init__.py**

```bash
touch src/agent/__init__.py tests/agent/__init__.py
```

- [ ] **Step 4: Implement protocol.py**

```python
# src/agent/protocol.py
import json
from dataclasses import dataclass
from typing import Any

VALID_ACTIONS = {"place_building", "dig", "cancel", "set_priority", "no_op"}


@dataclass
class StateMessage:
    cycle: int
    time: float
    data: dict[str, Any]


def parse_state_message(raw: str) -> StateMessage | None:
    """Parse a raw JSON line from the game. Returns StateMessage or None if not a state message."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if msg.get("type") != "state":
        return None

    return StateMessage(
        cycle=int(msg.get("cycle", 0)),
        time=float(msg.get("time", 0.0)),
        data=msg.get("data", {}),
    )


def build_action(action: str, **kwargs) -> dict[str, Any]:
    """Build an action command dict to send to the game."""
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown action: {action!r}. Valid: {VALID_ACTIONS}")
    return {"type": "action", "action": action, **kwargs}


def build_no_op() -> dict[str, Any]:
    return build_action("no_op")
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/agent/test_protocol.py -v
```
Expected: all 8 tests pass.

- [ ] **Step 6: Implement client.py**

```python
# src/agent/client.py
import asyncio
import json
import logging
from typing import AsyncIterator
from src.agent.protocol import parse_state_message, StateMessage, build_no_op

logger = logging.getLogger(__name__)


class BridgeClient:
    """Async TCP client that connects to the ONIBridge mod."""

    def __init__(self, host: str, port: int = 9999, reconnect_delay: float = 5.0):
        self.host = host
        self.port = port
        self.reconnect_delay = reconnect_delay
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self):
        """Connect (or reconnect) to the game bridge."""
        while True:
            try:
                self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
                logger.info(f"Connected to ONIBridge at {self.host}:{self.port}")
                return
            except (ConnectionRefusedError, OSError) as e:
                logger.warning(f"Connection failed: {e} — retrying in {self.reconnect_delay}s")
                await asyncio.sleep(self.reconnect_delay)

    async def state_stream(self) -> AsyncIterator[StateMessage]:
        """Yield StateMessage objects as they arrive from the game."""
        while True:
            try:
                line = await self._reader.readline()
                if not line:
                    logger.warning("Connection closed by game — reconnecting")
                    await self.connect()
                    continue
                raw = line.decode("utf-8").strip()
                if not raw:
                    continue
                msg = parse_state_message(raw)
                if msg is not None:
                    yield msg
            except (ConnectionResetError, BrokenPipeError):
                logger.warning("Connection reset — reconnecting")
                await self.connect()

    async def send_action(self, action: dict):
        """Send an action command to the game."""
        if self._writer is None:
            logger.error("Not connected — cannot send action")
            return
        try:
            line = json.dumps(action) + "\n"
            self._writer.write(line.encode("utf-8"))
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError) as e:
            logger.warning(f"Send failed: {e}")

    async def close(self):
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()


async def run_stub(host: str, port: int = 9999):
    """Stub runner — connect, print every state, send no_op each tick."""
    client = BridgeClient(host, port)
    await client.connect()

    async for state in client.state_stream():
        print(f"[Cycle {state.cycle}] resources={state.data.get('resources')} alerts={state.data.get('alerts')}")
        await client.send_action(build_no_op())


if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    asyncio.run(run_stub(host))
```

- [ ] **Step 7: Commit**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
git add src/agent/ tests/agent/
git commit -m "feat: Python stub agent with protocol parsing and TCP client"
```

---

## Task 8: End-to-End Round-Trip Verification

The moment of truth — stub client connects to the live mod, receives real game state, sends `no_op`, receives ack.

- [ ] **Step 1: Confirm ONI is running with the mod loaded on Ubuntu**

Check log for `[ONIBridge] Listening on port 9999`. ONI must be in-game (not at main menu) for `Game.Update` to fire and state to be sent.

- [ ] **Step 2: Run stub client from DGX Spark 1**

SSH to DGX Spark 1:
```bash
ssh nmyers@10.0.0.69
cd ~/Projects  # or wherever ONI-AI is cloned
git clone <repo-url> ONI-AI  # if not already cloned
cd ONI-AI
pip install -e . --quiet
python -m src.agent.client <ubuntu-ip>
```

Expected output (one line per ~10 game ticks):
```
Connected to ONIBridge at <ubuntu-ip>:9999
[Cycle 1] resources={'oxygen_kg': 45.2, 'water_kg': 120.0, ...} alerts=[]
[Cycle 1] resources={'oxygen_kg': 44.8, 'water_kg': 120.0, ...} alerts=[]
```

- [ ] **Step 3: Verify ACK appears in Ubuntu log**

In the tailed `Player.log` on Ubuntu, you should see:
```
[ONIBridge] ActionExecutor: no_op — ACK sent
```

- [ ] **Step 4: Send a place_building action manually to verify executor**

In a second terminal on DGX, use netcat to send a raw command:
```bash
UBUNTU_IP=<ubuntu-ip>
echo '{"type":"action","action":"place_building","building_id":"Outhouse","cell_x":5,"cell_y":5}' | nc $UBUNTU_IP 9999
```

Watch Ubuntu's `Player.log` for:
```
[ONIBridge] Placed Outhouse at (5,5)
```
And watch the game on screen — an outhouse construction order should appear at that cell. If `IsValidPlaceLocation` returns false, you'll see the rejection reason in the log instead.

- [ ] **Step 5: Commit verification note**

```bash
cd /Volumes/DevDrive-M4Pro/Projects/ONI-AI
git tag phase1-roundtrip-verified
git push origin main --tags
```

---

## Self-Review

**Spec coverage check:**
- [x] TCP server on port 9999 — Task 6
- [x] State snapshot: cycle, time, resources, duplicants, buildings, alerts — Task 4
- [x] Actions: place_building, dig, cancel, set_priority, no_op — Task 5
- [x] ACK messages — Task 3
- [x] Python client connects, receives state, sends action — Tasks 7-8
- [x] End-to-end round-trip verification — Task 8
- [x] Game DLLs in lib/ — Task 1

**Gaps addressed:**
- `grid_summary` (min/max oxygen, avg temperature) from the spec is deferred — it requires iterating 256x256 cells which needs a zone-based approach. Noted in Open Questions in the spec. Not blocking Phase 1.
- `cancel` action (cancel build order, not dig) not implemented in Task 5 — added alongside CancelDig which handles dig cancellation. Build order cancellation follows same pattern via `Deconstructable` component, add in Phase 2 if needed.

**Type consistency verified:**
- `StateMessage.data` is `dict[str, Any]` throughout
- `ActionCommand.Action` (C# string) maps to `action["action"]` (Python string) consistently
- `BridgeServer.SendAck()` signature matches `AckMessage` schema throughout
- `build_action()` kwargs pass through directly to dict — callers use correct kwarg names matching `ActionCommand` JSON properties
