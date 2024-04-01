namespace YoloDotNet.Models
{
    /// <summary>
    /// Model for holding extracted video metadata
    /// </summary>
    public class VideoMetaData
    {
        [JsonPropertyName("streams")]
        public VideoStream[] Streams { get; set; } = [];

        [JsonPropertyName("format")]
        public VideoFormat Format { get; set; } = new();
    }

    public class VideoFormat
    {
        [JsonPropertyName("bit_rate")]
        public string Bitrate { get; set; } = default!;
    }

    public class VideoStream
    {
        [JsonPropertyName("width")]
        public int Width { get; set; }

        [JsonPropertyName("height")]
        public int Height { get; set; }

        [JsonPropertyName("r_frame_rate")]
        public string Framerate { get; set; } = default!;

        [JsonPropertyName("duration")]
        public double Duration { get; set; }
    }
}
