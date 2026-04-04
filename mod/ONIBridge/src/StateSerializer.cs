using System;
using System.Collections.Generic;
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
                cycle      = TryGet("cycle",      GetCycle,      0),
                time       = TryGet("time",        GetTime,       0f),
                resources  = TryGet("resources",   GetResources,  (object)new {}),
                duplicants = TryGet("duplicants",  GetDuplicants, new List<object>()),
                buildings  = TryGet("buildings",   GetBuildings,  new List<object>()),
                alerts     = TryGet("alerts",      GetAlerts,     new List<string>()),
                tiles      = TryGet("tiles",       GetTiles,      (object)new {}),
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
            // Use WorldInventory with accessible=true to include all tracked resources.
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
                        foodKcal += edible.Calories;
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

        private static List<object> GetDuplicants()
        {
            var result = new List<object>();
            if (Components.MinionIdentities == null) return result;

            foreach (MinionIdentity minion in Components.MinionIdentities)
            {
                if (minion == null) continue;
                var pos = minion.transform.position;

                // StressMonitor.Instance has a public 'stress' field (0–1 range)
                float stress = 0f;
                var stressMon = minion.GetSMI<StressMonitor.Instance>();
                if (stressMon != null) stress = stressMon.stress.value;

                float health = 0f;
                var hp = minion.GetComponent<Health>();
                if (hp != null) health = hp.hitPoints;

                string currentTask = "idle";
                var chore = minion.GetComponent<ChoreDriver>()?.GetCurrentChore();
                if (chore != null) currentTask = chore.choreType?.Id ?? "unknown";

                result.Add(new
                {
                    id           = minion.GetInstanceID(),
                    name         = minion.name,
                    x            = (int)pos.x,
                    y            = (int)pos.y,
                    stress       = System.Math.Round(stress, 3),
                    health       = System.Math.Round(health, 1),
                    current_task = currentTask,
                });
            }
            return result;
        }

        private static List<object> GetBuildings()
        {
            var result = new List<object>();
            if (Components.BuildingCompletes == null) return result;

            foreach (BuildingComplete b in Components.BuildingCompletes)
            {
                if (b == null) continue;
                var pos = b.transform.position;
                var op  = b.GetComponent<Operational>();
                result.Add(new
                {
                    type        = b.Def?.PrefabID ?? "unknown",
                    x           = (int)pos.x,
                    y           = (int)pos.y,
                    operational = op != null && op.IsOperational,
                });
            }
            return result;
        }

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

        private static object GetTiles()
        {
            // Compute bounding box of all completed buildings
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
            int wx, wy;
            if (hasBuildings)
            {
                wx = minX - MARGIN;
                wy = minY - MARGIN;
            }
            else
            {
                // Fall back to 30×30 centered on world spawn
                wx = Grid.WidthInCells / 2 - 15;
                wy = Grid.HeightInCells / 2 - 15;
            }

            int ex = hasBuildings ? maxX + MARGIN : wx + 30;
            int ey = hasBuildings ? maxY + MARGIN : wy + 30;

            // Clamp to world bounds
            wx = System.Math.Max(0, wx);
            wy = System.Math.Max(0, wy);
            ex = System.Math.Min(Grid.WidthInCells - 1, ex);
            ey = System.Math.Min(Grid.HeightInCells - 1, ey);

            int w = ex - wx + 1;
            int h = ey - wy + 1;

            var data = new System.Collections.Generic.List<object>();
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
    }
}
