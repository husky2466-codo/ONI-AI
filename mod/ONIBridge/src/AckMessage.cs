using Newtonsoft.Json;

namespace ONIBridge
{
    public class AckMessage
    {
        [JsonProperty("type")]
        public string Type { get; } = "ack";

        [JsonProperty("action")]
        public string Action { get; set; } = "";

        [JsonProperty("success")]
        public bool Success { get; set; }

        [JsonProperty("error")]
        public string? Error { get; set; }
    }
}
