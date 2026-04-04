using System.IO;
using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// One-shot autoload: on first ready tick, reads autoload.txt next to the mod DLL.
    /// If the file exists, loads the specified save and deletes the file immediately.
    /// Written by the Python runner before restarting the game to set the training save.
    /// </summary>
    public static class AutoloadConfig
    {
        private static readonly string ConfigPath = Path.Combine(
            Application.persistentDataPath,
            "mods", "Dev", "ONIBridge", "autoload.txt"
        );

        /// <summary>
        /// Called from BridgeTicker once per session after world is confirmed loaded.
        /// Reads autoload.txt, loads the save, then deletes the file (one-shot).
        /// </summary>
        public static void TryAutoload()
        {
            if (!File.Exists(ConfigPath)) return;

            string savePath = File.ReadAllText(ConfigPath).Trim();
            File.Delete(ConfigPath);

            if (string.IsNullOrEmpty(savePath)) return;

            if (!File.Exists(savePath))
            {
                Debug.LogWarning($"[ONIBridge] autoload.txt pointed to missing save: {savePath}");
                return;
            }

            Debug.Log($"[ONIBridge] Autoloading save: {savePath}");
            SaveLoader.Instance.Load(savePath);
        }
    }
}
