# Extended Game State Schema — Design Spec

**Date:** 2026-04-04
**Status:** Approved for implementation

---

## Overview

The current ONIBridge state payload is a shallow MVP sufficient for the first live session but
inadequate for real colony management. This spec defines the complete game state schema the
Harmony mod must serialize to give the AI full visibility into every system it needs to make
correct decisions.

Changes are additive — no existing fields are removed or renamed.

**Implementation note:** All new C# sections follow the existing `TryGet` pattern in
`StateSerializer.cs`. If a section's API throws (world not loaded, component missing), it
returns an empty fallback. The Python agent treats missing or null fields as "data not yet
available" and degrades gracefully.

---

## Priority Levels

| Priority | Systems | Rationale |
|----------|---------|-----------|
| P0 | Pending actions, dupe full stats, printing pod | Stops known bugs; gates all strategic decisions |
| P1 | Storage inventory, machine state, research | Required for blueprint task board prerequisite layer |
| P2 | Power networks, rooms | Prevents mid-game failures |
| P3 | Atmosphere zones, temperature zones | Mid-game optimization |
| P4 | Germs, plants, critters | Late-game / farming phase |

---

## Update Frequency & Token Cost

The full schema is large. Naively including every field in the Gemini prompt every tick will
balloon token cost and degrade reasoning quality. Fields must be classified by update frequency:

| Frequency | Fields | Rationale |
|-----------|--------|-----------|
| Every tick | `duplicants`, `resources`, `alerts`, `colony`, `tiles`, `perimeter` | Change every tick; agent needs current values to act |
| On change only | `buildings`, `storage`, `printing_pod`, `research`, `power_networks`, `rooms` | Change rarely; sending every tick wastes tokens |

**Implementation:** `runner.py` maintains a shadow copy of the last-sent slow-update fields.
Each tick it diffs against the current state. If a slow field changed, it is included in the
prompt. If unchanged, it is omitted and replaced with a one-line summary:
`"[storage: unchanged since cycle 8]"` or `"[research: MedicineI in progress, 45/100pts]"`.

The full payload is always written to the episode JSONL log regardless of prompt inclusion.
This keeps training data complete while keeping prompt tokens manageable.

**Expected prompt size with this approach:** ~800–1200 tokens per tick in normal operation,
vs. ~3000–5000 if all fields are included every tick.

---

## P0: Pending Actions (Python-Side — No C# Changes)

See Spatial Perimeter System spec. This is tracked in `runner.py` and injected into the prompt.
No mod changes required.

---

## P0: Duplicants — Full Stats

### Current gaps
- No organic vs. bionic type
- No hunger, bladder, stamina, morale
- No skills or traits
- No equipment/suit state

### C# Implementation

Replace the current `GetDuplicants()` body in `StateSerializer.cs`:

```csharp
private static List<object> GetDuplicants()
{
    var result = new List<object>();
    if (Components.MinionIdentities == null) return result;

    foreach (MinionIdentity minion in Components.MinionIdentities)
    {
        if (minion == null) continue;
        var pos = minion.transform.position;

        // --- Type detection ---
        // Bionic dupes have a RobotBatteryMonitor (verify via decompile)
        bool isBionic = minion.GetSMI<RobotBatteryMonitor.Instance>() != null;

        // --- Shared stats ---
        float stress = 0f;
        var stressMon = minion.GetSMI<StressMonitor.Instance>();
        if (stressMon != null) stress = System.Math.Clamp(stressMon.stress.value, 0f, 1f);

        float health = 0f;
        var hp = minion.GetComponent<Health>();
        if (hp != null) health = hp.hitPoints;

        string currentTask = "idle";
        int taskX = -1, taskY = -1;
        var chore = minion.GetComponent<ChoreDriver>()?.GetCurrentChore();
        if (chore != null)
        {
            currentTask = chore.choreType?.Id ?? "unknown";
            var loc = chore.target?.GetComponent<KMonoBehaviour>()?.transform?.position;
            if (loc.HasValue) { taskX = (int)loc.Value.x; taskY = (int)loc.Value.y; }
        }

        // --- Skills ---
        var skills = new Dictionary<string, int>();
        var attrs = minion.GetComponent<Attributes>();
        if (attrs != null)
        {
            foreach (var attrId in new[]{ "Digging","Building","Cooking","Learning",
                                           "Caring","Athletics","Strength","Art",
                                           "Ranching","Farming","Machinery" })
            {
                var attr = attrs.Get(attrId);
                if (attr != null) skills[attrId.ToLower()] = (int)attr.GetTotalValue();
            }
        }

        // --- Traits ---
        var traits = new List<string>();
        var traitComp = minion.GetComponent<Traits>();
        if (traitComp != null)
            foreach (var t in traitComp.TraitList)
                if (t != null) traits.Add(t.Id);

        // --- Equipment ---
        string equipment = "none";
        var eq = minion.GetComponent<MinionEquipmentSlot>();
        // Verify exact API via decompile; fallback to "none" if unavailable

        object typeData;
        if (isBionic)
        {
            float charge = 0f;
            bool charging = false;
            var bat = minion.GetSMI<RobotBatteryMonitor.Instance>();
            // Verify RobotBatteryMonitor API via decompile
            typeData = new { type = "bionic", charge_pct = charge, charging = charging };
        }
        else
        {
            // --- Organic needs ---
            float hunger = 0f, bladder = 0f, stamina = 0f;
            var calories = minion.GetSMI<CalorieMonitor.Instance>();
            if (calories != null) hunger = System.Math.Clamp(calories.calories / 1000f, 0f, 1f);

            var bladderMon = minion.GetSMI<BladderMonitor.Instance>();
            if (bladderMon != null) bladder = bladderMon.bladder.value;

            var fatigue = minion.GetSMI<StaminaMonitor.Instance>();
            if (fatigue != null) stamina = fatigue.stamina.value;

            // Morale: sum of all morale modifiers
            int morale = 0;
            var moraleAttr = attrs?.Get("QualityOfLife");
            if (moraleAttr != null) morale = (int)moraleAttr.GetTotalValue();

            typeData = new
            {
                type     = "organic",
                hunger   = System.Math.Round(hunger, 3),
                bladder  = System.Math.Round(bladder, 3),
                stamina  = System.Math.Round(stamina, 3),
                morale   = morale,
            };
        }

        result.Add(new
        {
            id            = minion.GetInstanceID(),
            name          = minion.name,
            x             = (int)pos.x,
            y             = (int)pos.y,
            stress        = System.Math.Round(stress, 3),
            health        = System.Math.Round(health, 1),
            current_task  = currentTask,
            task_x        = taskX,
            task_y        = taskY,
            skills        = skills,
            traits        = traits,
            equipment     = equipment,
            dupe_data     = typeData,
        });
    }
    return result;
}
```

### JSON Output

```json
{
  "id": -126192,
  "name": "Lindsay",
  "x": 132, "y": 203,
  "stress": 0.12,
  "health": 100.0,
  "current_task": "Dig",
  "task_x": 130, "task_y": 201,
  "skills": { "digging": 4, "building": 2, "cooking": 0, "learning": 1 },
  "traits": ["Twinkletoes", "NightOwl"],
  "equipment": "none",
  "dupe_data": {
    "type": "organic",
    "hunger": 0.65,
    "bladder": 0.2,
    "stamina": 0.88,
    "morale": 12
  }
}
```

For a bionic dupe:
```json
{
  "dupe_data": { "type": "bionic", "charge_pct": 0.74, "charging": false }
}
```

**Stress fix:** Note `Math.Clamp(stress, 0f, 1f)` is applied above. This fixes the known bug
where `StressMonitor.stress.value` returned values > 1.0.

---

## P0: Printing Pod

The printing pod is one of the highest-leverage decision points in the game. The AI must know
when a decision is pending and what the options are.

### C# Implementation

Add `GetPrintingPod()` to `StateSerializer.cs` and add `printing_pod` to `Serialize()`.

```csharp
private static object GetPrintingPod()
{
    // Immigration.Instance manages the printing pod schedule.
    // Verify exact field names via decompile of Assembly-CSharp.dll.
    var immigration = Immigration.Instance;
    if (immigration == null) return new { status = "unavailable" };

    // timeBeforeSpawn: seconds until next print offer (verify field name)
    float timeRemaining = immigration.timeBeforeSpawn;
    float cycleDuration = 600f; // seconds per cycle at 1x speed
    float cyclesRemaining = timeRemaining / cycleDuration;

    // Check if immigrants are currently waiting for a decision
    // ImmigrantPerks or similar — verify via decompile
    bool waitingForDecision = false;
    var offers = new List<object>();

    // The immigration system spawns immigrants into a care package or dupe
    // when the timer hits 0. The exact API for reading current offers
    // requires decompile verification. Implement as best-effort:
    // if immigration.HasImmigrant() or similar returns true, read the offers.

    // Fallback shape if offers can't be read:
    return new
    {
        status               = waitingForDecision ? "waiting_for_decision" : "cooldown",
        cycles_until_next    = System.Math.Round(cyclesRemaining, 1),
        offers               = offers,  // empty list if not yet readable
    };
}
```

**Implementation note:** The printing pod API is the most uncertain in this spec. The implementer
must decompile `Assembly-CSharp.dll` in Rider and search for `Immigration`, `ImmigrantPerks`,
`CarePackage`, and `MinionStartingStats` to find the exact offer-reading API. The shape below
is the target output once found.

### JSON Output (target)

```json
{
  "printing_pod": {
    "status": "waiting_for_decision",
    "cycles_until_next": 0.0,
    "offers": [
      {
        "type": "duplicant",
        "subtype": "organic",
        "name": "Otto",
        "traits": ["Twinkletoes", "Mouth Breather"],
        "skills": { "digging": 5, "building": 1 }
      },
      {
        "type": "care_package",
        "contents": [{ "element": "Water", "mass_kg": 2000 }]
      },
      {
        "type": "duplicant",
        "subtype": "bionic",
        "name": "Axiom",
        "traits": ["PowerHungry"],
        "skills": { "machinery": 3, "building": 2 }
      }
    ]
  }
}
```

### New Action: `accept_print`

Add to `ActionExecutor.cs` and `protocol.py`:

```
accept_print: { type, action, offer_index }   // 0, 1, or 2
```

The action calls the appropriate immigration acceptance API (verify via decompile).

---

## P1: Storage Inventory

Required by the perimeter task board's `PrerequisiteResolver`.

### C# Implementation

Add `GetStorage()` to `StateSerializer.cs`:

```csharp
private static List<object> GetStorage()
{
    var result = new List<object>();

    // Walk all Storage components in the world
    foreach (Storage storage in Components.Storages)
    {
        if (storage == null || storage.gameObject == null) continue;

        var building = storage.GetComponent<BuildingComplete>();
        // Only serialize named buildings (skip internal/hidden storages)
        if (building == null) continue;

        var pos = storage.transform.position;
        float capacity = storage.capacityKg;

        var contents = new List<object>();
        foreach (GameObject item in storage.items)
        {
            if (item == null) continue;
            var pe = item.GetComponent<PrimaryElement>();
            if (pe == null) continue;
            contents.Add(new
            {
                element  = pe.Element?.id.ToString() ?? "Unknown",
                mass_kg  = System.Math.Round(pe.Mass / 1000f, 2),
            });
        }

        result.Add(new
        {
            building_id = building.Def?.PrefabID ?? "unknown",
            x           = (int)pos.x,
            y           = (int)pos.y,
            capacity_kg = System.Math.Round(capacity / 1000f, 1),
            contents    = contents,
        });
    }
    return result;
}
```

### JSON Output

```json
[
  {
    "building_id": "StorageLocker",
    "x": 15, "y": 22,
    "capacity_kg": 20.0,
    "contents": [
      { "element": "IronOre", "mass_kg": 0.45 },
      { "element": "Sandstone", "mass_kg": 0.12 }
    ]
  },
  {
    "building_id": "Refrigerator",
    "x": 10, "y": 18,
    "capacity_kg": 0.2,
    "contents": [
      { "element": "MushBar", "mass_kg": 0.003 }
    ]
  }
]
```

---

## P1: Machine State

The current buildings list only reports `operational: true/false`. Each machine needs internal
state to know if it's starving for inputs or backing up on outputs.

### C# Implementation

Modify `GetBuildings()` to add machine state fields:

```csharp
private static List<object> GetBuildings()
{
    var result = new List<object>();
    if (Components.BuildingCompletes == null) return result;

    foreach (BuildingComplete b in Components.BuildingCompletes)
    {
        if (b == null) continue;
        var pos = b.transform.position;
        var op = b.GetComponent<Operational>();
        bool isOp = op != null && op.IsOperational;

        // Machine internals
        bool isWorking = op != null && op.IsActive;
        float progressPct = 0f;
        var workable = b.GetComponent<Workable>();
        // Verify: some machines use WorkTime/WorkTimeRemaining or similar
        // Fallback to 0 if not applicable

        // Input/output storage attached to this building
        var inputContents  = new List<object>();
        var outputContents = new List<object>();

        // Buildings have one or more Storage components; first is usually input, last output.
        // This heuristic works for most machines; verify for edge cases.
        var storages = b.GetComponents<Storage>();
        if (storages != null && storages.Length > 0)
        {
            var inputStorage = storages[0];
            foreach (GameObject item in inputStorage.items)
            {
                if (item == null) continue;
                var pe = item.GetComponent<PrimaryElement>();
                if (pe == null) continue;
                inputContents.Add(new
                {
                    element = pe.Element?.id.ToString() ?? "Unknown",
                    mass_kg = System.Math.Round(pe.Mass / 1000f, 3),
                });
            }

            if (storages.Length > 1)
            {
                var outputStorage = storages[storages.Length - 1];
                foreach (GameObject item in outputStorage.items)
                {
                    if (item == null) continue;
                    var pe = item.GetComponent<PrimaryElement>();
                    if (pe == null) continue;
                    outputContents.Add(new
                    {
                        element = pe.Element?.id.ToString() ?? "Unknown",
                        mass_kg = System.Math.Round(pe.Mass / 1000f, 3),
                    });
                }
            }
        }

        result.Add(new
        {
            type            = b.Def?.PrefabID ?? "unknown",
            x               = (int)pos.x,
            y               = (int)pos.y,
            operational     = isOp,
            working         = isWorking,
            progress_pct    = System.Math.Round(progressPct, 1),
            input_contents  = inputContents,
            output_contents = outputContents,
        });
    }
    return result;
}
```

### JSON Output

```json
{
  "type": "RockCrusher",
  "x": 20, "y": 15,
  "operational": true,
  "working": true,
  "progress_pct": 42.0,
  "input_contents":  [{ "element": "IronOre",     "mass_kg": 0.4 }],
  "output_contents": [{ "element": "RefinedMetal", "mass_kg": 0.1 }]
}
```

---

## P1: Research

### C# Implementation

Add `GetResearch()` to `StateSerializer.cs`:

```csharp
private static object GetResearch()
{
    var db = Db.Get();
    if (db == null) return new { unlocked = new List<string>() };

    var unlocked = new List<string>();
    string currentTech = null;
    float currentProgress = 0f;
    float currentCost = 0f;

    foreach (Tech tech in db.Techs.resources)
    {
        if (tech == null) continue;
        var techInstance = Research.Instance?.GetTechProgress(tech);
        if (techInstance == null) continue;

        if (techInstance.IsComplete())
        {
            unlocked.Add(tech.Id);
        }
        else if (Research.Instance?.activeResearch?.tech == tech)
        {
            currentTech     = tech.Id;
            // Verify exact progress fields via decompile
            // currentProgress = techInstance.progressInventory.??
            currentCost     = tech.costsByResearchTypeID?.Values?.Sum() ?? 0f;
        }
    }

    return new
    {
        unlocked          = unlocked,
        current_tech      = currentTech,
        current_progress  = System.Math.Round(currentProgress, 1),
        current_cost      = System.Math.Round(currentCost, 0),
    };
}
```

### JSON Output

```json
{
  "research": {
    "unlocked": ["BasicResearch", "FarmingTech", "ImprovedOxygen"],
    "current_tech": "MedicineI",
    "current_progress": 45.0,
    "current_cost": 100.0
  }
}
```

---

## P2: Power Networks

### C# Implementation

Add `GetPowerNetworks()` to `StateSerializer.cs`:

```csharp
private static List<object> GetPowerNetworks()
{
    var result = new List<object>();
    // Verify: Circuit/ElectricalUtility API via decompile
    // Known: Game.Instance.electricalConduitSystem or CircuitManager
    var circuitManager = Game.Instance?.electricalConduitSystem;
    if (circuitManager == null) return result;

    // Iterate circuits — verify exact API
    // Each circuit has: Generators, Consumers, Batteries
    // This is a best-effort implementation; verify field names via decompile

    return result;  // Implement after decompile verification
}
```

**Implementation note:** Power network internals require decompile verification.
Search for `ElectricalUtility`, `CircuitManager`, `Generator.IsProducingPower()` (already
used in `GetResources()`). The existing `Components.Generators` iteration in `GetResources()`
shows the pattern; expand it per-circuit.

### JSON Output (target)

```json
{
  "power_networks": [
    {
      "circuit_id": 1,
      "total_generation_w": 800,
      "total_consumption_w": 420,
      "net_w": 380,
      "overloaded": false,
      "batteries": [
        { "type": "Battery", "x": 18, "y": 20, "charge_pct": 0.72 }
      ]
    }
  ]
}
```

---

## P2: Rooms

### C# Implementation

Add `GetRooms()` to `StateSerializer.cs`:

```csharp
private static List<object> GetRooms()
{
    var result = new List<object>();
    // Verify: RoomProber.Instance and Room API via decompile
    // Known pattern: RoomProber.Instance exists, Room has .roomType, .cavity
    var prober = RoomProber.Instance;
    if (prober == null) return result;

    // Iterate rooms — verify exact API
    // Each Room: roomType.Id (e.g. "Bedroom"), cavity.minX/Y/maxX/Y, requirements

    return result;  // Implement after decompile verification
}
```

### JSON Output (target)

```json
{
  "rooms": [
    {
      "type": "Bedroom",
      "bounds": { "x1": 110, "y1": 198, "x2": 118, "y2": 202 },
      "requirements_met": true,
      "missing": []
    },
    {
      "type": "MessHall",
      "bounds": { "x1": 119, "y1": 198, "x2": 127, "y2": 202 },
      "requirements_met": false,
      "missing": ["MessTable"]
    }
  ]
}
```

---

## P3: Atmosphere Zones

Rather than per-cell atmosphere (too large), report per-room summaries.

### JSON Output (target)

```json
{
  "atmosphere_zones": [
    {
      "zone": "main_base",
      "bounds": { "x1": 108, "y1": 196, "x2": 140, "y2": 210 },
      "o2_kg": 4.2,
      "co2_kg": 0.3,
      "avg_temp_c": 24.1,
      "breathable": true
    }
  ]
}
```

Derive zones from room cavities (reuse Room data). No additional C# required beyond rooms.

---

## P4: Germs, Plants, Critters

These are deferred to a later phase. The diagnostic system (`ColonyDiagnosticUtility`,
already used in `GetAlerts()`) surfaces disease alerts, which is sufficient for P0-P2.

Detailed germ/plant/critter serialization is planned for the farming and disease management
phases of agent development.

---

## Colony Summary (New Top-Level Field)

Add a `colony` summary field to `Serialize()` for quick AI context:

```csharp
private static object GetColonySummary()
{
    int total = 0, organic = 0, bionic = 0;
    if (Components.MinionIdentities != null)
    {
        foreach (MinionIdentity m in Components.MinionIdentities)
        {
            if (m == null) continue;
            total++;
            bool isBionic = m.GetSMI<RobotBatteryMonitor.Instance>() != null;
            if (isBionic) bionic++; else organic++;
        }
    }

    return new
    {
        dupe_count     = total,
        organic_count  = organic,
        bionic_count   = bionic,
        o2_needed_gs   = organic * 0.1f,    // 100g O2/dupe/second
        charge_needed_w = bionic * 120,     // ~120W per charging bionic (verify)
    };
}
```

### JSON Output

```json
{
  "colony": {
    "dupe_count": 3,
    "organic_count": 3,
    "bionic_count": 0,
    "o2_needed_gs": 0.3,
    "charge_needed_w": 0
  }
}
```

---

## Updated `Serialize()` Method

```csharp
public static object Serialize()
{
    return new
    {
        cycle           = TryGet("cycle",          GetCycle,          0),
        time            = TryGet("time",           GetTime,           0f),
        colony          = TryGet("colony",         GetColonySummary,  (object)new {}),
        resources       = TryGet("resources",      GetResources,      (object)new {}),
        duplicants      = TryGet("duplicants",     GetDuplicants,     new List<object>()),
        buildings       = TryGet("buildings",      GetBuildings,      new List<object>()),
        storage         = TryGet("storage",        GetStorage,        new List<object>()),
        printing_pod    = TryGet("printing_pod",   GetPrintingPod,    (object)null),
        research        = TryGet("research",       GetResearch,       (object)new {}),
        power_networks  = TryGet("power_networks", GetPowerNetworks,  new List<object>()),
        rooms           = TryGet("rooms",          GetRooms,          new List<object>()),
        alerts          = TryGet("alerts",         GetAlerts,         new List<string>()),
        tiles           = TryGet("tiles",          GetTiles,          (object)new {}),
        perimeter       = TryGet("perimeter",      GetPerimeter,      (object)null),
    };
}
```

---

## Files Changed

| File | Change |
|------|--------|
| `mod/ONIBridge/src/StateSerializer.cs` | Add GetColonySummary, extend GetDuplicants, add GetStorage, GetPrintingPod, GetResearch, GetPowerNetworks, GetRooms, GetPerimeter; extend GetBuildings with machine state |
| `mod/ONIBridge/src/ActionExecutor.cs` | Add `accept_print` handler |
| `src/agent/protocol.py` | Add `accept_print` action type |
| `src/agent/llm.py` | Update `_format_state()` to include new fields in prompt |

---

## Decompile Checklist

The following APIs require verification against `Assembly-CSharp.dll` before implementation:

- [ ] `RobotBatteryMonitor.Instance` — bionic charge level
- [ ] `Immigration.Instance` fields — printing pod offer reading
- [ ] `Research.Instance.GetTechProgress()` — research progress fields
- [ ] `CircuitManager` / `ElectricalUtility` — power circuit iteration
- [ ] `RoomProber.Instance` — room iteration API
- [ ] `CalorieMonitor.Instance.calories` — hunger level field name
- [ ] `StaminaMonitor.Instance.stamina` — stamina field name
- [ ] `BladderMonitor.Instance.bladder` — bladder field name

All other APIs used above are already proven in the existing `StateSerializer.cs`.
