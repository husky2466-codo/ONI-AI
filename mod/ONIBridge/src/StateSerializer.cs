using System;
using System.Collections.Generic;
using Klei.AI;
using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// Serializes the current ONI game state into a JSON-safe object
    /// that gets sent to the AI agent each tick.
    /// </summary>
    public static class StateSerializer
    {
        // Known diagnostic IDs used by ColonyDiagnosticUtility.GetDiagnostic(id, worldId)
        private static readonly string[] DiagnosticIds = new[]
        {
            "BreathabilityDiagnostic",
            "FoodDiagnostic",
            "StressDiagnostic",
            "BedDiagnostic",
            "ToiletDiagnostic",
            "IdleDiagnostic",
            "HeatDiagnostic",
            "EntombedDiagnostic",
            "DecorDiagnostic",
            "FloatingRocketDiagnostic",
        };

        public static object Serialize()
        {
            return new
            {
                cycle          = TryGet("cycle",          GetCycle,          0),
                time           = TryGet("time",            GetTime,           0f),
                resources      = TryGet("resources",       GetResources,      (object)new {}),
                duplicants     = TryGet("duplicants",      GetDuplicants,     new List<object>()),
                buildings      = TryGet("buildings",       GetBuildings,      new List<object>()),
                storage        = TryGet("storage",         GetStorage,        new List<object>()),
                printing_pod   = TryGet("printing_pod",    GetPrintingPod,    (object)new {}),
                research       = TryGet("research",        GetResearch,       (object)new {}),
                power_networks = TryGet("power_networks",  GetPowerNetworks,  new List<object>()),
                rooms          = TryGet("rooms",           GetRooms,          new List<object>()),
                alerts         = TryGet("alerts",          GetAlerts,         new List<string>()),
                tiles          = TryGet("tiles",           GetTiles,          (object)new {}),
                perimeter      = TryGet("perimeter",       GetPerimeter,      (object)null),
            };
        }

        private static T TryGet<T>(string name, System.Func<T> fn, T fallback)
        {
            try   { return fn(); }
            catch (Exception ex)
            {
                Debug.LogWarning($"[ONIBridge] {name} failed: {ex.Message}");
                return fallback;
            }
        }

        // ------------------------------------------------------------------ //

        private static int GetCycle()
        {
            // GetCycle() is 0-indexed internally; add 1 to match the in-game display.
            return GameClock.Instance != null ? (int)GameClock.Instance.GetCycle() + 1 : 0;
        }

        private static float GetTime()
        {
            return GameClock.Instance != null ? GameClock.Instance.GetTime() : 0f;
        }

        private static object GetResources()
        {
            WorldInventory inv = null;
            if (ClusterManager.Instance != null)
                inv = ClusterManager.Instance.activeWorld?.worldInventory;

            float oxygen = 0f, water = 0f, co2 = 0f;
            if (inv != null)
            {
                oxygen = inv.GetAmount(SimHashes.Oxygen.CreateTag(), true);
                water  = inv.GetAmount(SimHashes.Water.CreateTag(), true);
                co2    = inv.GetAmount(SimHashes.CarbonDioxide.CreateTag(), true);
            }

            // Food in storage: sum all Edible components across the world.
            float foodKcal = 0f;
            if (Components.Edibles != null)
            {
                foreach (Edible edible in Components.Edibles)
                {
                    if (edible != null)
                        foodKcal += edible.Calories / 1000f;  // internal unit is kcal*1000
                }
            }

            // Power: sum wattage of all currently producing generators.
            float powerWatts = 0f;
            if (Components.Generators != null)
            {
                foreach (Generator gen in Components.Generators)
                {
                    if (gen != null && gen.IsProducingPower())
                        powerWatts += gen.WattageRating;
                }
            }

            return new
            {
                oxygen_kg   = oxygen     / 1000f,
                water_kg    = water      / 1000f,
                food_kcal   = foodKcal,
                power_kw    = powerWatts / 1000f,
                co2_kg      = co2        / 1000f,
            };
        }

        // ------------------------------------------------------------------ //
        // P0: Full duplicant stats

        private static List<object> GetDuplicants()
        {
            var result = new List<object>();
            if (Components.MinionIdentities == null) return result;

            foreach (MinionIdentity minion in Components.MinionIdentities)
            {
                if (minion == null) continue;
                var pos = minion.transform.position;

                // --- Type detection ---
                // Bionic dupes have a RobotBatteryMonitor SMI (verify via decompile)
                bool isBionic = minion.GetSMI<RobotBatteryMonitor.Instance>() != null;

                // --- Stress (clamped) ---
                float stress = 0f;
                var stressMon = minion.GetSMI<StressMonitor.Instance>();
                if (stressMon != null)
                {
                    stress = stressMon.stress.value;
                    if (stress < 0f) stress = 0f;
                    if (stress > 1f) stress = 1f;
                }

                // --- Health ---
                float health = 0f;
                var hp = minion.GetComponent<Health>();
                if (hp != null) health = hp.hitPoints;

                // --- Current task + target location ---
                string currentTask = "idle";
                int taskX = -1, taskY = -1;
                var chore = minion.GetComponent<ChoreDriver>()?.GetCurrentChore();
                if (chore != null)
                {
                    currentTask = chore.choreType?.Id ?? "unknown";
                    try
                    {
                        var loc = chore.target?.GetComponent<KMonoBehaviour>()?.transform?.position;
                        if (loc.HasValue) { taskX = (int)loc.Value.x; taskY = (int)loc.Value.y; }
                    }
                    catch { /* task location unavailable */ }
                }

                // --- Skills (attribute levels via Klei.AI.Attributes extension) ---
                var skills = new Dictionary<string, int>();
                try
                {
                    var attrs = minion.GetAttributes();
                    if (attrs != null)
                    {
                        foreach (var attr in attrs.AttributeTable)
                        {
                            if (attr == null) continue;
                            float val = attr.GetTotalValue();
                            if (val > 0f)
                                skills[attr.Id] = (int)val;
                        }
                    }
                }
                catch { /* attributes unavailable */ }

                // --- Traits (Klei.AI.Traits component) ---
                var traits = new List<string>();
                try
                {
                    var traitComp = minion.GetComponent<Klei.AI.Traits>();
                    if (traitComp != null)
                    {
                        foreach (string tid in traitComp.GetTraitIds())
                            traits.Add(tid);
                    }
                }
                catch { /* traits unavailable */ }

                // --- Type-specific data ---
                object typeData;
                if (isBionic)
                {
                    float charge = 0f;
                    try
                    {
                        var bat = minion.GetSMI<RobotBatteryMonitor.Instance>();
                        if (bat != null && bat.amountInstance != null)
                        {
                            float max = bat.amountInstance.GetMax();
                            if (max > 0f)
                                charge = bat.amountInstance.value / max;
                        }
                    }
                    catch { /* bionic battery unavailable */ }
                    typeData = new { type = "bionic", charge_pct = System.Math.Round(charge, 3) };
                }
                else
                {
                    float hunger = 0f, bladder = 0f, stamina = 0f;
                    int morale = 0;

                    // CalorieMonitor: calories is AmountInstance; use .value / GetMax() for 0-1 range
                    try
                    {
                        var calMon = minion.GetSMI<CalorieMonitor.Instance>();
                        if (calMon != null && calMon.calories != null)
                        {
                            float max = calMon.calories.GetMax();
                            if (max > 0f)
                                hunger = 1f - (calMon.calories.value / max); // 0=full, 1=starving
                        }
                    }
                    catch { }

                    // BladderMonitor: bladder field is private; use Db.Get().Amounts.Bladder.Lookup
                    try
                    {
                        var bladderAmt = Db.Get().Amounts.Bladder.Lookup(minion.gameObject);
                        if (bladderAmt != null)
                            bladder = bladderAmt.value / 100f; // 0-1 range (max is 100)
                    }
                    catch { }

                    try
                    {
                        var fatigue = minion.GetSMI<StaminaMonitor.Instance>();
                        if (fatigue != null) stamina = fatigue.stamina.value;
                    }
                    catch { }

                    // Morale via QualityOfLife attribute — decompile needed for Attributes component
                    // try
                    // {
                    //     var moraleAttr = minion.GetComponent<AttributeConverters>()...
                    //     if (moraleAttr != null) morale = (int)moraleAttr.GetTotalValue();
                    // }
                    // catch { }

                    typeData = new
                    {
                        type    = "organic",
                        hunger  = System.Math.Round(hunger, 3),
                        bladder = System.Math.Round(bladder, 3),
                        stamina = System.Math.Round(stamina, 3),
                        morale  = morale,
                    };
                }

                result.Add(new
                {
                    id           = minion.GetInstanceID(),
                    name         = minion.name,
                    x            = (int)pos.x,
                    y            = (int)pos.y,
                    stress       = System.Math.Round(stress, 3),
                    health       = System.Math.Round(health, 1),
                    current_task = currentTask,
                    task_x       = taskX,
                    task_y       = taskY,
                    skills       = skills,
                    traits       = traits,
                    dupe_data    = typeData,
                });
            }
            return result;
        }

        // ------------------------------------------------------------------ //
        // P1: Buildings with machine state

        private static List<object> GetBuildings()
        {
            var result = new List<object>();
            if (Components.BuildingCompletes == null) return result;

            foreach (BuildingComplete b in Components.BuildingCompletes)
            {
                if (b == null) continue;
                var pos = b.transform.position;
                var op  = b.GetComponent<Operational>();
                bool isOp     = op != null && op.IsOperational;
                bool isActive = op != null && op.IsActive;

                // Input/output storage from machine's own Storage components
                var inputContents  = new List<object>();
                var outputContents = new List<object>();
                try
                {
                    var storages = b.GetComponents<Storage>();
                    if (storages != null && storages.Length > 0)
                    {
                        foreach (GameObject item in storages[0].items)
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
                            foreach (GameObject item in storages[storages.Length - 1].items)
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
                }
                catch { /* storage unavailable for this building */ }

                result.Add(new
                {
                    type            = b.Def?.PrefabID ?? "unknown",
                    x               = (int)pos.x,
                    y               = (int)pos.y,
                    operational     = isOp,
                    working         = isActive,
                    input_contents  = inputContents,
                    output_contents = outputContents,
                });
            }
            return result;
        }

        // ------------------------------------------------------------------ //
        // P1: Storage containers

        private static List<object> GetStorage()
        {
            var result = new List<object>();
            // Components.Storages may not exist — use FindObjectsOfType as fallback
            // DECOMPILE: verify whether Components.Storages exists in ONI's Components registry
            Storage[] storages;
            try { storages = UnityEngine.Object.FindObjectsOfType<Storage>(); }
            catch { return result; }
            foreach (Storage storage in storages)
            {
                if (storage == null || storage.gameObject == null) continue;

                // Only serialize buildings with named Storage containers
                var building = storage.GetComponent<BuildingComplete>();
                if (building == null) continue;

                var pos = storage.transform.position;
                var contents = new List<object>();

                try
                {
                    foreach (GameObject item in storage.items)
                    {
                        if (item == null) continue;
                        var pe = item.GetComponent<PrimaryElement>();
                        if (pe == null) continue;
                        contents.Add(new
                        {
                            element = pe.Element?.id.ToString() ?? "Unknown",
                            mass_kg = System.Math.Round(pe.Mass / 1000f, 2),
                        });
                    }
                }
                catch { /* storage contents unavailable */ }

                result.Add(new
                {
                    building_id = building.Def?.PrefabID ?? "unknown",
                    x           = (int)pos.x,
                    y           = (int)pos.y,
                    capacity_kg = System.Math.Round(storage.capacityKg / 1000f, 1),
                    contents    = contents,
                });
            }
            return result;
        }

        // ------------------------------------------------------------------ //
        // P0: Printing pod

        private static object GetPrintingPod()
        {
            var immigration = Immigration.Instance;
            if (immigration == null) return new { status = "unavailable" };

            // timeBeforeSpawn: seconds until next print offer
            // Verify field name via decompile — best-effort below
            float timeRemaining = 0f;
            try { timeRemaining = immigration.timeBeforeSpawn; }
            catch { }

            float cycleDuration = 600f; // seconds per cycle at 1x speed
            float cyclesRemaining = timeRemaining / cycleDuration;

            // Offer detection — requires decompile verification.
            // Immigration.Instance.HasImmigrant() or similar.
            // For now, return a best-effort status.
            bool waitingForDecision = false;
            var offers = new List<object>();
            try
            {
                // This will be expanded after decompile confirms the API.
                // For now surface the timer so AI knows when to expect a decision.
                waitingForDecision = immigration.ImmigrantsAvailable;
            }
            catch { }

            return new
            {
                status            = waitingForDecision ? "waiting_for_decision" : "cooldown",
                cycles_until_next = System.Math.Round(cyclesRemaining, 1),
                offers            = offers,
            };
        }

        // ------------------------------------------------------------------ //
        // P1: Research

        private static object GetResearch()
        {
            var unlocked = new List<string>();
            string currentTech = null;
            float currentProgress = 0f;

            var research = Research.Instance;
            if (research == null) return new { unlocked, current_tech = currentTech, current_progress = currentProgress };

            // Enumerate all techs to find completed ones
            try
            {
                foreach (Tech tech in Db.Get().Techs.resources)
                {
                    if (tech == null) continue;
                    var ti = research.GetTechInstance(tech.Id);
                    if (ti != null && ti.IsComplete())
                        unlocked.Add(tech.Id);
                }
            }
            catch { }

            // Active research
            try
            {
                var active = research.GetActiveResearch();
                if (active != null)
                {
                    currentTech     = active.tech?.Id;
                    currentProgress = active.GetTotalPercentageComplete();
                }
            }
            catch { }

            return new
            {
                unlocked,
                current_tech     = currentTech,
                current_progress = System.Math.Round(currentProgress, 3),
            };
        }

        // ------------------------------------------------------------------ //
        // P2: Power networks (stub — requires decompile to implement fully)

        private static List<object> GetPowerNetworks()
        {
            var result = new List<object>();
            // Full implementation requires decompile of ElectricalUtility/CircuitManager.
            // Returning empty list until decompile verification is complete.
            // The existing Components.Generators sum in GetResources() provides a
            // basic power overview until this is implemented.
            return result;
        }

        // ------------------------------------------------------------------ //
        // P2: Rooms (stub — requires decompile to implement fully)

        private static List<object> GetRooms()
        {
            var result = new List<object>();
            // Full implementation requires decompile of RoomProber.Instance and Room API.
            // Returning empty list until decompile verification is complete.
            return result;
        }

        // ------------------------------------------------------------------ //
        // Alerts

        private static List<string> GetAlerts()
        {
            var alerts = new List<string>();
            try
            {
                var util = ColonyDiagnosticUtility.Instance;
                if (util == null) return alerts;
                if (ClusterManager.Instance == null) return alerts;

                int worldId = ClusterManager.Instance.activeWorldId;

                foreach (string diagId in DiagnosticIds)
                {
                    try
                    {
                        var diag = util.GetDiagnostic(diagId, worldId);
                        if (diag?.LatestResult == null) continue;

                        var opinion = diag.LatestResult.opinion;
                        if (opinion == ColonyDiagnostic.DiagnosticResult.Opinion.Warning ||
                            opinion == ColonyDiagnostic.DiagnosticResult.Opinion.Bad ||
                            opinion == ColonyDiagnostic.DiagnosticResult.Opinion.DuplicantThreatening)
                        {
                            alerts.Add($"{diagId}: {diag.LatestResult.Message}");
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning($"[ONIBridge] Alert {diagId} failed: {ex.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[ONIBridge] GetAlerts failed: {ex.Message}");
            }
            return alerts;
        }

        // ------------------------------------------------------------------ //
        // Tiles (follows active perimeter)

        private static object GetTiles()
        {
            int wx, wy, ex, ey;

            // If a perimeter is active, center the tile window on it (+ 5 tile padding)
            var perim = PerimeterManager.Active;
            if (perim != null)
            {
                const int PAD = 5;
                wx = perim.X1 - PAD;
                wy = perim.Y1 - PAD;
                ex = perim.X2 + PAD;
                ey = perim.Y2 + PAD;
            }
            else
            {
                // Default: bounding box of all completed buildings + 15 tile margin
                int minX = int.MaxValue, maxX = int.MinValue;
                int minY = int.MaxValue, maxY = int.MinValue;
                bool hasBuildings = false;

                if (Components.BuildingCompletes != null)
                {
                    foreach (BuildingComplete b in Components.BuildingCompletes)
                    {
                        if (b == null) continue;
                        var pos = b.transform.position;
                        int bx = (int)pos.x, by = (int)pos.y;
                        if (bx < minX) minX = bx;
                        if (bx > maxX) maxX = bx;
                        if (by < minY) minY = by;
                        if (by > maxY) maxY = by;
                        hasBuildings = true;
                    }
                }

                const int MARGIN = 15;
                if (hasBuildings)
                {
                    wx = minX - MARGIN;
                    wy = minY - MARGIN;
                    ex = maxX + MARGIN;
                    ey = maxY + MARGIN;
                }
                else
                {
                    wx = Grid.WidthInCells / 2 - 15;
                    wy = Grid.HeightInCells / 2 - 15;
                    ex = wx + 30;
                    ey = wy + 30;
                }
            }

            // Clamp to world bounds
            wx = System.Math.Max(0, wx);
            wy = System.Math.Max(0, wy);
            ex = System.Math.Min(Grid.WidthInCells - 1, ex);
            ey = System.Math.Min(Grid.HeightInCells - 1, ey);

            int w = ex - wx + 1;
            int h = ey - wy + 1;

            // Cap window to 64×64 to prevent unbounded payload growth on large colonies
            const int MAX_WINDOW = 64;
            if (w > MAX_WINDOW)
            {
                int midX = wx + w / 2;
                wx = midX - MAX_WINDOW / 2;
                ex = wx + MAX_WINDOW - 1;
                w = MAX_WINDOW;
            }
            if (h > MAX_WINDOW)
            {
                int midY = wy + h / 2;
                wy = midY - MAX_WINDOW / 2;
                ey = wy + MAX_WINDOW - 1;
                h = MAX_WINDOW;
            }

            if (w <= 0 || h <= 0)
            {
                Debug.LogWarning($"[ONIBridge] GetTiles: degenerate window w={w} h={h}, skipping");
                return new { x = 0, y = 0, w = 0, h = 0, data = new List<object>() };
            }

            var data = new List<object>();
            for (int row = 0; row < h; row++)
            {
                for (int col = 0; col < w; col++)
                {
                    int cx = wx + col;
                    int cy = wy + row;
                    int cell = Grid.XYToCell(cx, cy);
                    if (!Grid.IsValidCell(cell))
                    {
                        data.Add(new object[] { "Invalid", 0f });
                        continue;
                    }
                    string elementName = Grid.Element[cell]?.id.ToString() ?? "Vacuum";
                    float mass = Grid.Mass[cell];
                    data.Add(new object[] { elementName, System.Math.Round(mass, 1) });
                }
            }

            return new { x = wx, y = wy, w, h, data };
        }

        // ------------------------------------------------------------------ //
        // Perimeter

        private static object GetPerimeter()
        {
            return PerimeterManager.Serialize();
        }
    }
}
