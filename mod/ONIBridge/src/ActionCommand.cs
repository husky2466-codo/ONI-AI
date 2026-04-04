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

        // Duplicant assignment
        [JsonProperty("duplicant_id")]
        public int DuplicantId { get; set; } = -1;

        [JsonProperty("skill")]
        public string? Skill { get; set; }
    }
}
