using System.Collections.Generic;
using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// Serializes the current ONI game state into a JSON-safe object
    /// that gets sent to the AI agent each tick.
    /// Expands as we map more of the game's internal APIs.
    /// </summary>
    public static class StateSerializer
    {
        public static object Serialize()
        {
            return new
            {
                cycle = GetCycle(),
                time = GetTime(),
                resources = GetResources(),
                duplicants = GetDuplicants(),
                alerts = GetAlerts(),
            };
        }

        private static int GetCycle()
        {
            // TODO: GameClock.Instance.GetCycle()
            return 0;
        }

        private static float GetTime()
        {
            // TODO: GameClock.Instance.GetTime()
            return 0f;
        }

        private static object GetResources()
        {
            // TODO: Read from WorldInventory.Instance or ClusterManager
            return new
            {
                oxygen_kg = 0f,
                water_kg = 0f,
                food_kcal = 0f,
                power_kw = 0f,
            };
        }

        private static List<object> GetDuplicants()
        {
            var result = new List<object>();
            // TODO: foreach (MinionIdentity minion in Components.MinionIdentities)
            // {
            //     result.Add(new {
            //         id = minion.GetInstanceID(),
            //         name = minion.name,
            //         x = (int)minion.transform.position.x,
            //         y = (int)minion.transform.position.y,
            //         stress = minion.GetComponent<StressMonitor.Instance>()?.stress ?? 0f,
            //         health = minion.GetComponent<Health>()?.hitPoints ?? 0f,
            //     });
            // }
            return result;
        }

        private static List<string> GetAlerts()
        {
            var alerts = new List<string>();
            // TODO: Read from Notifier or ColonyDiagnosticUtility
            return alerts;
        }
    }
}
