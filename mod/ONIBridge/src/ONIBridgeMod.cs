using HarmonyLib;
using KMod;
using System.Collections.Generic;
using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// Entry point for the ONI Bridge mod.
    /// Starts the TCP bridge server and creates a persistent GameObject
    /// with a BridgeTicker coroutine — no Harmony patches needed for the tick loop.
    /// </summary>
    public class ONIBridgeMod : UserMod2
    {
        public const int DEFAULT_PORT = 9999;

        public override void OnLoad(Harmony harmony)
        {
            base.OnLoad(harmony);
            Debug.Log("[ONIBridge] Mod loaded — starting bridge server...");
            BridgeServer.Instance.Start(DEFAULT_PORT);

            // Create a persistent GameObject to host the BridgeTicker coroutine.
            // DontDestroyOnLoad ensures it survives scene transitions.
            var go = new GameObject("ONIBridgeTicker");
            go.AddComponent<BridgeTicker>();
            Object.DontDestroyOnLoad(go);
            Debug.Log("[ONIBridge] BridgeTicker created.");
        }

        public override void OnAllModsLoaded(Harmony harmony, IReadOnlyList<Mod> mods)
        {
            base.OnAllModsLoaded(harmony, mods);
            Debug.Log("[ONIBridge] All mods loaded.");
        }
    }
}
