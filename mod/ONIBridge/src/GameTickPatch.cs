using HarmonyLib;
using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// Hooks into the game's update loop to:
    ///   1. Drain pending AI action commands each frame (main thread safe)
    ///   2. Send a state snapshot to the connected agent every N sim ticks
    /// </summary>
    [HarmonyPatch(typeof(Game), "Update")]
    public static class GameTickPatch
    {
        // Send state every 10 game updates (~1 second at normal speed)
        private const int STATE_INTERVAL = 10;
        private static int _tickCounter = 0;

        [HarmonyPostfix]
        public static void Postfix()
        {
            var server = BridgeServer.Instance;

            // Always drain actions — keeps latency low
            server.DrainActions();

            // Throttle state snapshots
            _tickCounter++;
            if (_tickCounter < STATE_INTERVAL) return;
            _tickCounter = 0;

            if (!server.IsConnected) return;

            try
            {
                var state = StateSerializer.Serialize();
                server.SendState(state);
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"[ONIBridge] State serialization failed: {ex.Message}");
            }
        }
    }
}
