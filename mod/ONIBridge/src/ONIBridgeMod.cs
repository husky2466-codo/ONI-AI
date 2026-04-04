using HarmonyLib;
using KMod;
using System.Collections.Generic;
using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// Entry point for the ONI Bridge mod.
    /// Starts a WebSocket server inside the game process so an external
    /// AI agent can read game state and issue action commands.
    /// </summary>
    public class ONIBridgeMod : UserMod2
    {
        public const int DEFAULT_PORT = 9999;

        public override void OnLoad(Harmony harmony)
        {
            base.OnLoad(harmony);
            Debug.Log("[ONIBridge] Mod loaded — starting bridge server...");
            BridgeServer.Instance.Start(DEFAULT_PORT);
        }

        public override void OnAllModsLoaded(Harmony harmony, IReadOnlyList<Mod> mods)
        {
            base.OnAllModsLoaded(harmony, mods);
            Debug.Log("[ONIBridge] All mods loaded.");
        }
    }
}
