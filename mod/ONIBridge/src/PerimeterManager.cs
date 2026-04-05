using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// Stores all active build zones (max 5), sorted by priority descending.
    /// No game sim interaction — dupes cannot see or interact with zones.
    /// Completion detection is handled Python-side; Python sends abandon_perimeter(id)
    /// when auto-complete triggers or the agent explicitly abandons.
    /// </summary>
    public static class PerimeterManager
    {
        private static readonly List<PerimeterData> _zones = new List<PerimeterData>();

        public static IReadOnlyList<PerimeterData> Zones => _zones;

        /// <summary>Highest-priority zone — tile window follows this one.</summary>
        public static PerimeterData Focused => _zones.Count > 0 ? _zones[0] : null;

        /// <summary>
        /// Place a new zone. Returns "placed" on success, or a rejection reason string.
        /// </summary>
        public static string Place(string id, int x1, int y1, int x2, int y2, string goal, int priority)
        {
            if (_zones.Count >= 5)
            {
                Debug.LogWarning("[ONIBridge] place_perimeter rejected — zone cap (5) reached");
                return "rejected_zone_cap";
            }
            _zones.Add(new PerimeterData
            {
                Id = id,
                X1 = x1, Y1 = y1, X2 = x2, Y2 = y2,
                Goal = goal,
                Priority = priority,
            });
            _zones.Sort((a, b) => b.Priority.CompareTo(a.Priority));
            Debug.Log($"[ONIBridge] Zone placed: {id} goal={goal} priority={priority} bounds=({x1},{y1})-({x2},{y2})");
            return "placed";
        }

        /// <summary>Abandon a specific zone by id.</summary>
        public static void Abandon(string id)
        {
            int removed = _zones.RemoveAll(z => z.Id == id);
            if (removed > 0)
                Debug.Log($"[ONIBridge] Zone abandoned: {id}");
            else
                Debug.LogWarning($"[ONIBridge] abandon_perimeter: zone {id} not found");
        }

        /// <summary>Produces the zones array for the state message.</summary>
        public static object Serialize()
        {
            return _zones.Select(z => (object)new
            {
                id       = z.Id,
                goal     = z.Goal,
                bounds   = new { x1 = z.X1, y1 = z.Y1, x2 = z.X2, y2 = z.Y2 },
                priority = z.Priority,
                status   = "active",
            }).ToList();
        }
    }

    public class PerimeterData
    {
        public string Id;
        public int X1, Y1, X2, Y2;
        public string Goal;
        public int Priority;
    }
}
