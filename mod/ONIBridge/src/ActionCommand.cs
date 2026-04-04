using Newtonsoft.Json;

namespace ONIBridge
{
    /// <summary>
    /// Represents an action command sent from the AI agent to the game.
    /// All commands arrive as newline-delimited JSON over the TCP socket.
    /// </summary>
    public class ActionCommand
    {
        [JsonProperty("type")]
        public string Type { get; set; } = "action";

        [JsonProperty("action")]
        public string Action { get; set; } = "";

        // Building placement
        [JsonProperty("building_id")]
        public string? BuildingId { get; set; }

        // Dig / build target cell
        [JsonProperty("cell_x")]
        public int CellX { get; set; }

        [JsonProperty("cell_y")]
        public int CellY { get; set; }

        // Priority (1-9)
        [JsonProperty("priority")]
        public int Priority { get; set; } = 5;

        // Game speed: 0=paused, 1=normal, 2=fast, 3=ultra
        [JsonProperty("speed")]
        public int? Speed { get; set; }

        // Perimeter bounding box
        [JsonProperty("x1")]
        public int X1 { get; set; }

        [JsonProperty("y1")]
        public int Y1 { get; set; }

        [JsonProperty("x2")]
        public int X2 { get; set; }

        [JsonProperty("y2")]
        public int Y2 { get; set; }

        // Perimeter goal (e.g. "oxygen_production")
        [JsonProperty("goal")]
        public string? Goal { get; set; }

        // Research assignment
        [JsonProperty("tech_id")]
        public string? TechId { get; set; }

        // Building enable/disable toggle
        [JsonProperty("enabled")]
        public bool? Enabled { get; set; }

        // Printing pod offer index (0, 1, or 2)
        [JsonProperty("offer_index")]
        public int? OfferIndex { get; set; }

        // Duplicant assignment
        [JsonProperty("duplicant_id")]
        public int DuplicantId { get; set; } = -1;

        [JsonProperty("skill")]
        public string? Skill { get; set; }

        // Helper accessors used by ActionExecutor
        public int GetInt(string field)
        {
            switch (field)
            {
                case "x1": return X1;
                case "y1": return Y1;
                case "x2": return X2;
                case "y2": return Y2;
                case "cell_x": return CellX;
                case "cell_y": return CellY;
                case "priority": return Priority;
                case "offer_index": return OfferIndex ?? 0;
                default: return 0;
            }
        }

        public string? GetString(string field)
        {
            switch (field)
            {
                case "goal": return Goal;
                case "tech_id": return TechId;
                case "building_id": return BuildingId;
                default: return null;
            }
        }

        public bool GetBool(string field)
        {
            switch (field)
            {
                case "enabled": return Enabled ?? true;
                default: return false;
            }
        }
    }
}
