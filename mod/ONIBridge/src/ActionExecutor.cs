using UnityEngine;

namespace ONIBridge
{
    /// <summary>
    /// Executes AI action commands on the main Unity thread by calling
    /// ONI's internal C# methods directly — same code path as the UI.
    /// Must only be called from the main thread (via BridgeServer.DrainActions).
    /// </summary>
    public static class ActionExecutor
    {
        public static void Execute(ActionCommand cmd)
        {
            try
            {
                switch (cmd.Action)
                {
                    case "place_building":
                        PlaceBuilding(cmd);
                        break;
                    case "dig":
                        Dig(cmd);
                        break;
                    case "cancel_dig":
                        CancelDig(cmd);
                        break;
                    case "set_priority":
                        SetPriority(cmd);
                        break;
                    case "no_op":
                        break;
                    default:
                        Debug.LogWarning($"[ONIBridge] Unknown action: {cmd.Action}");
                        break;
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[ONIBridge] ActionExecutor failed ({cmd.Action}): {ex.Message}");
            }
        }

        private static void PlaceBuilding(ActionCommand cmd)
        {
            // TODO: Implement using BuildingDef.TryPlace() or PlayerController
            // BuildingDef def = Assets.GetBuildingDef(cmd.BuildingId);
            // int cell = Grid.XYToCell(cmd.CellX, cmd.CellY);
            // def.TryPlace(null, cell, Orientation.Neutral, ...);
            Debug.Log($"[ONIBridge] PlaceBuilding: {cmd.BuildingId} at ({cmd.CellX},{cmd.CellY}) — stub");
        }

        private static void Dig(ActionCommand cmd)
        {
            // TODO: Implement using DigTool or Chore system
            // int cell = Grid.XYToCell(cmd.CellX, cmd.CellY);
            // Game.Instance.userMenu... or direct chore creation
            Debug.Log($"[ONIBridge] Dig at ({cmd.CellX},{cmd.CellY}) — stub");
        }

        private static void CancelDig(ActionCommand cmd)
        {
            Debug.Log($"[ONIBridge] CancelDig at ({cmd.CellX},{cmd.CellY}) — stub");
        }

        private static void SetPriority(ActionCommand cmd)
        {
            Debug.Log($"[ONIBridge] SetPriority at ({cmd.CellX},{cmd.CellY}) p={cmd.Priority} — stub");
        }
    }
}
