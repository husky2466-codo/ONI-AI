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
                cycle      = GetCycle(),
                time       = GetTime(),
                resources  = GetResources(),
                duplicants = GetDuplicants(),
                buildings  = GetBuildings(),
                alerts     = GetAlerts(),
            };
        }

        // ------------------------------------------------------------------ //

        private static int GetCycle()
        {
            return GameClock.Instance != null ? (int)GameClock.Instance.GetCycle() : 0;
        }

        private static float GetTime()
        {
            return GameClock.Instance != null ? GameClock.Instance.GetTime() : 0f;
        }

        private static object GetResources()
        {
            // WorldInventory is accessed through ClusterManager -> activeWorld -> worldInventory
            WorldInventory inv = null;
            if (ClusterManager.Instance != null)
                inv = ClusterManager.Instance.activeWorld?.worldInventory;

            float oxygen = 0f, water = 0f, co2 = 0f;
            if (inv != null)
            {
                oxygen = inv.GetAmount(SimHashes.Oxygen.CreateTag(), false);
                water  = inv.GetAmount(SimHashes.Water.CreateTag(), false);
                co2    = inv.GetAmount(SimHashes.CarbonDioxide.CreateTag(), false);
            }

            // RationTracker.Get() returns the singleton; GetAmountConsumed() gives today's
            // calories consumed (no "remaining" API exists).
            float foodConsumed = 0f;
            var rt = RationTracker.Get();
            if (rt != null) foodConsumed = rt.GetAmountConsumed();

            // Sum rated wattage across all active generators.
            // Note: WattageRating is nameplate capacity, not real-time output.
            // A generator running below rated capacity (e.g. low fuel) will overstate actual power.
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
                oxygen_kg         = oxygen       / 1000f,
                water_kg          = water        / 1000f,
                // food_kcal_today: calories consumed today, NOT food in storage.
                // RationTracker has no "remaining food" API; use buildings[] for storage buildings.
                food_kcal_today   = foodConsumed,
                power_kw          = powerWatts   / 1000f,
                co2_kg            = co2          / 1000f,
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
            var util   = ColonyDiagnosticUtility.Instance;
            if (util == null) return alerts;
            if (ClusterManager.Instance == null) return alerts;

            int worldId = ClusterManager.Instance.activeWorldId;

            // Query each known diagnostic by string ID.
            // GetDiagnostic returns null if the diagnostic isn't registered for this world.
            foreach (string diagId in DiagnosticIds)
            {
                var diag = util.GetDiagnostic(diagId, worldId);
                if (diag?.LatestResult == null) continue;

                var opinion = diag.LatestResult.opinion;
                // Report Bad and DuplicantThreatening (most severe).
                // Opinion ordering (least to most severe): Good, Normal, Acceptable, Warning, Bad, DuplicantThreatening.
                // Good/Normal/Acceptable/Warning are intentionally excluded as non-alert-worthy.
                if (opinion == ColonyDiagnostic.DiagnosticResult.Opinion.Bad ||
                    opinion == ColonyDiagnostic.DiagnosticResult.Opinion.DuplicantThreatening)
                {
                    alerts.Add($"{diagId}: {diag.LatestResult.Message}");
                }
            }
            return alerts;
        }
    }
}
