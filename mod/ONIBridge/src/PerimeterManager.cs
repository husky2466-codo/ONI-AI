using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// Mod-side only. Stores the single active perimeter bounding box.
    /// No game sim interaction — dupes cannot see or interact with it.
    /// Completion detection is handled Python-side; Python sends abandon_perimeter
    /// when auto-complete triggers.
    /// </summary>
    public static class PerimeterManager
    {
        public static PerimeterData Active { get; private set; } = null;

        /// <summary>
        /// Place a new perimeter. Returns false if one is already active.
        /// </summary>
        public static bool Place(int x1, int y1, int x2, int y2, string goal)
        {
            if (Active != null)
            {
                Debug.LogWarning("[ONIBridge] place_perimeter rejected — perimeter already active");
                return false;
            }
            Active = new PerimeterData
            {
                Id     = System.Guid.NewGuid().ToString("N").Substring(0, 8),
                X1     = x1,
                Y1     = y1,
                X2     = x2,
                Y2     = y2,
                Goal   = goal,
                Status = "active",
            };
            Debug.Log($"[ONIBridge] Perimeter placed: {Active.Id} goal={goal} bounds=({x1},{y1})-({x2},{y2})");
            return true;
        }

        /// <summary>
        /// Abandon the active perimeter (auto-complete or explicit abandon).
        /// </summary>
        public static void Abandon()
        {
            if (Active != null)
                Debug.Log($"[ONIBridge] Perimeter abandoned: {Active.Id}");
            Active = null;
        }

        /// <summary>
        /// Produces the perimeter payload for the state message. Returns null when no perimeter active.
        /// </summary>
        public static object Serialize()
        {
            if (Active == null) return null;
            return new
            {
                id     = Active.Id,
                goal   = Active.Goal,
                bounds = new { x1 = Active.X1, y1 = Active.Y1, x2 = Active.X2, y2 = Active.Y2 },
                status = Active.Status,
            };
        }
    }

    public class PerimeterData
    {
        public string Id;
        public int X1, Y1, X2, Y2;
        public string Goal;
        public string Status;  // "active" | "abandoned"
    }
}
