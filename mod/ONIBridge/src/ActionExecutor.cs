using System;
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
                        BridgeServer.Instance.SendAck(cmd.Action, true);
                        break;
                    case "place_perimeter":
                    {
                        string id = cmd.GetString("id") ?? System.Guid.NewGuid().ToString("N").Substring(0, 8);
                        int x1 = cmd.GetInt("x1"), y1 = cmd.GetInt("y1");
                        int x2 = cmd.GetInt("x2"), y2 = cmd.GetInt("y2");
                        string goal = cmd.GetString("goal") ?? "unknown";
                        int priority = cmd.GetInt("priority");
                        if (priority < 1 || priority > 9) priority = 5;
                        string result = PerimeterManager.Place(id, x1, y1, x2, y2, goal, priority);
                        if (result == "placed")
                            BridgeServer.Instance.SendAck(cmd.Action, true);
                        else
                            BridgeServer.Instance.SendAck(cmd.Action, false, result);
                        break;
                    }
                    case "abandon_perimeter":
                    {
                        string id = cmd.GetString("id") ?? "";
                        if (!string.IsNullOrEmpty(id))
                            PerimeterManager.Abandon(id);
                        BridgeServer.Instance.SendAck(cmd.Action, true);
                        break;
                    }
                    case "assign_research":
                    {
                        string techId = cmd.GetString("tech_id");
                        if (string.IsNullOrEmpty(techId))
                        {
                            BridgeServer.Instance.SendAck(cmd.Action, false, "tech_id is required");
                            break;
                        }
                        var research = Research.Instance;
                        if (research == null)
                        {
                            BridgeServer.Instance.SendAck(cmd.Action, false, "Research.Instance unavailable");
                            break;
                        }
                        Tech tech = null;
                        try { tech = Db.Get().Techs.Get(techId); } catch { }
                        if (tech == null)
                        {
                            BridgeServer.Instance.SendAck(cmd.Action, false, $"Unknown tech: {techId}");
                            break;
                        }
                        var ti = research.GetTechInstance(techId);
                        if (ti != null && ti.IsComplete())
                        {
                            BridgeServer.Instance.SendAck(cmd.Action, false, "tech_already_complete");
                            break;
                        }
                        research.SetActiveResearch(tech);
                        BridgeServer.Instance.SendAck(cmd.Action, true);
                        Debug.Log($"[ONIBridge] Active research set to {techId}");
                        break;
                    }
                    case "enable_building":
                    {
                        int cell = Grid.XYToCell(cmd.CellX, cmd.CellY);
                        if (!Grid.IsValidCell(cell))
                        {
                            BridgeServer.Instance.SendAck(cmd.Action, false, $"Invalid cell ({cmd.CellX},{cmd.CellY})");
                            break;
                        }
                        var go = Grid.Objects[cell, (int)ObjectLayer.Building];
                        if (go == null)
                        {
                            BridgeServer.Instance.SendAck(cmd.Action, false, "No building at cell");
                            break;
                        }
                        var toggle = go.GetComponent<BuildingEnabledButton>();
                        if (toggle == null)
                        {
                            BridgeServer.Instance.SendAck(cmd.Action, false, "Building does not support enable/disable");
                            break;
                        }
                        bool enabled = cmd.GetBool("enabled");
                        toggle.IsEnabled = enabled;
                        BridgeServer.Instance.SendAck(cmd.Action, true);
                        Debug.Log($"[ONIBridge] Building at ({cmd.CellX},{cmd.CellY}) enabled={enabled}");
                        break;
                    }
                    case "accept_print":
                    {
                        int offerIndex = cmd.GetInt("offer_index");
                        if (offerIndex < 0 || offerIndex > 2)
                        {
                            BridgeServer.Instance.SendAck(cmd.Action, false, "offer_index_out_of_range");
                            break;
                        }
                        var immigration = Immigration.Instance;
                        if (immigration == null)
                        {
                            BridgeServer.Instance.SendAck(cmd.Action, false, "immigration_unavailable");
                            break;
                        }
                        // Check if offer is available — field name to be verified via decompile
                        bool hasOffer = false;
                        try { hasOffer = immigration.ImmigrantsAvailable; }
                        catch { }
                        if (!hasOffer)
                        {
                            BridgeServer.Instance.SendAck(cmd.Action, false, "error_no_active_offer");
                            break;
                        }
                        // DECOMPILE REQUIRED: Immigration.AcceptImmigrant method not yet verified.
                        // Signal Python side to use xdotool fallback until confirmed.
                        BridgeServer.Instance.SendAck(cmd.Action, false, "xdotool_required:accept_print_needs_decompile");
                        break;
                    }
                    case "set_speed":
                    {
                        int speed = cmd.Speed ?? 1;
                        speed = System.Math.Max(0, System.Math.Min(3, speed));
                        try
                        {
                            if (SpeedControlScreen.Instance == null)
                            {
                                BridgeServer.Instance.SendAck(cmd.Action, false, "SpeedControlScreen not available");
                                break;
                            }
                            if (speed == 0)
                                SpeedControlScreen.Instance.Pause();
                            else
                                SpeedControlScreen.Instance.SetSpeed(speed);
                            BridgeServer.Instance.SendAck(cmd.Action, true);
                            Debug.Log($"[ONIBridge] Speed set to {speed}");
                        }
                        catch (System.Exception ex)
                        {
                            Debug.LogWarning($"[ONIBridge] set_speed failed: {ex.Message}");
                            BridgeServer.Instance.SendAck(cmd.Action, false, ex.Message);
                        }
                        break;
                    }
                    default:
                        Debug.LogWarning($"[ONIBridge] Unknown action: {cmd.Action}");
                        BridgeServer.Instance.SendAck(cmd.Action, false, $"Unknown action: {cmd.Action}");
                        break;
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[ONIBridge] ActionExecutor failed ({cmd.Action}): {ex.Message}");
                BridgeServer.Instance.SendAck(cmd.Action, false, ex.Message);
            }
        }

        private static void PlaceBuilding(ActionCommand cmd)
        {
            if (string.IsNullOrEmpty(cmd.BuildingId))
            {
                BridgeServer.Instance.SendAck(cmd.Action, false, "building_id is required");
                return;
            }

            var def = Assets.GetBuildingDef(cmd.BuildingId);
            if (def == null)
            {
                BridgeServer.Instance.SendAck(cmd.Action, false, $"Unknown building: {cmd.BuildingId}");
                return;
            }

            int cell = Grid.XYToCell(cmd.CellX, cmd.CellY);
            if (!Grid.IsValidCell(cell))
            {
                BridgeServer.Instance.SendAck(cmd.Action, false, $"Invalid cell ({cmd.CellX},{cmd.CellY})");
                return;
            }

            string reason;
            if (!def.IsValidPlaceLocation(null, cell, Orientation.Neutral, out reason))
            {
                BridgeServer.Instance.SendAck(cmd.Action, false, $"Cannot place: {reason}");
                return;
            }

            // DefaultElements() returns Tag[] with the default construction materials.
            // TryPlace takes a world-space Vector3 position, not a cell int.
            var selectedElements = def.DefaultElements();
            def.TryPlace(
                null,
                Grid.CellToPosCBC(cell, Grid.SceneLayer.Building),
                Orientation.Neutral,
                selectedElements,
                0
            );
            BridgeServer.Instance.SendAck(cmd.Action, true);
            Debug.Log($"[ONIBridge] Placed {cmd.BuildingId} at ({cmd.CellX},{cmd.CellY})");
        }

        private static void Dig(ActionCommand cmd)
        {
            int cell = Grid.XYToCell(cmd.CellX, cmd.CellY);
            if (!Grid.IsValidCell(cell))
            {
                BridgeServer.Instance.SendAck(cmd.Action, false, $"Invalid cell ({cmd.CellX},{cmd.CellY})");
                return;
            }

            if (!Grid.IsSolidCell(cell))
            {
                BridgeServer.Instance.SendAck(cmd.Action, false, "Cell is not solid — nothing to dig");
                return;
            }

            var digPlacer = Grid.Objects[cell, (int)ObjectLayer.DigPlacer];
            if (digPlacer == null)
            {
                // TryGetPrefab(Tag) returns GameObject directly (single-arg overload).
                // Util.KInstantiate(GameObject, GameObject parent, string name) is the
                // Klei utility that clones with name; position is set afterwards.
                var prefab = Assets.TryGetPrefab(new Tag("DigPlacer"));
                if (prefab == null)
                {
                    BridgeServer.Instance.SendAck(cmd.Action, false, "DigPlacer prefab not found");
                    return;
                }

                var go = Util.KInstantiate(prefab, null, "DigPlacer");
                go.transform.position = Grid.CellToPosCBC(cell, Grid.SceneLayer.Building);
                go.SetActive(true);
                // Register in the grid so duplicate dig commands on the same cell are detected.
                Grid.Objects[cell, (int)ObjectLayer.DigPlacer] = go;
            }

            BridgeServer.Instance.SendAck(cmd.Action, true);
            Debug.Log($"[ONIBridge] Dig queued at ({cmd.CellX},{cmd.CellY})");
        }

        private static void CancelDig(ActionCommand cmd)
        {
            int cell = Grid.XYToCell(cmd.CellX, cmd.CellY);
            if (!Grid.IsValidCell(cell))
            {
                BridgeServer.Instance.SendAck(cmd.Action, false, $"Invalid cell ({cmd.CellX},{cmd.CellY})");
                return;
            }

            var digPlacer = Grid.Objects[cell, (int)ObjectLayer.DigPlacer];
            if (digPlacer != null)
            {
                // Use KDestroyGameObject so Klei's OnCleanUp hooks fire and the chore
                // scheduler dequeues the dig errand cleanly. Raw Object.Destroy skips this.
                Util.KDestroyGameObject(digPlacer);
                BridgeServer.Instance.SendAck(cmd.Action, true);
            }
            else
            {
                BridgeServer.Instance.SendAck(cmd.Action, false, "No dig order at cell");
            }
        }

        private static void SetPriority(ActionCommand cmd)
        {
            int cell = Grid.XYToCell(cmd.CellX, cmd.CellY);
            if (!Grid.IsValidCell(cell))
            {
                BridgeServer.Instance.SendAck(cmd.Action, false, $"Invalid cell ({cmd.CellX},{cmd.CellY})");
                return;
            }

            // Clamp to valid ONI priority range 1–9 (System.Math.Clamp not in net471)
            int priority = System.Math.Max(1, System.Math.Min(9, cmd.Priority));

            // Check Building layer first, then DigPlacer (cancel-dig orders are also prioritizable)
            var go = Grid.Objects[cell, (int)ObjectLayer.Building]
                  ?? Grid.Objects[cell, (int)ObjectLayer.DigPlacer];

            if (go == null)
            {
                BridgeServer.Instance.SendAck(cmd.Action, false, "No prioritizable object at cell");
                return;
            }

            var prioritizable = go.GetComponent<Prioritizable>();
            if (prioritizable == null)
            {
                BridgeServer.Instance.SendAck(cmd.Action, false, "Object is not prioritizable");
                return;
            }

            prioritizable.SetMasterPriority(new PrioritySetting(PriorityScreen.PriorityClass.basic, priority));
            BridgeServer.Instance.SendAck(cmd.Action, true);
            Debug.Log($"[ONIBridge] Priority set to {priority} at ({cmd.CellX},{cmd.CellY})");
        }
    }
}
