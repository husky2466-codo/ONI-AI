using System.Collections;
using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// MonoBehaviour that runs a coroutine to drain actions and send state
    /// every STATE_INTERVAL_SECONDS real-time seconds.
    /// Attached to a persistent GameObject created at mod load time.
    /// Does not depend on any ONI-specific tick hooks.
    /// </summary>
    public class BridgeTicker : MonoBehaviour
    {
        private const float STATE_INTERVAL_SECONDS = 1.0f;

        private IEnumerator Start()
        {
            Debug.Log("[ONIBridge] BridgeTicker coroutine started.");
            while (true)
            {
                yield return new WaitForSeconds(STATE_INTERVAL_SECONDS);

                var server = BridgeServer.Instance;
                server.DrainActions();

                if (!server.IsConnected) continue;

                // Skip if the game world isn't loaded yet
                if (GameClock.Instance == null || ClusterManager.Instance == null) continue;

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
}
