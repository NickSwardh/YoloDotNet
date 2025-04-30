namespace YoloDotNet.Models
{
    public record VideoMetadata(
        int Width,
        int Height,
        int TargetWidth,
        int TargetHeight,
        double Duration,
        double FPS,
        double TargetFPS,
        long TotalFrames,
        long TargetTotalFrames);

    internal class Metadata
    {
        [JsonPropertyName("width")]
        public int Width { get; set; }

        [JsonPropertyName("height")]
        public int Height { get; set; }

        [JsonPropertyName("frameratenumerator")]
        public int FrameRateNumerator { get; set; }

        [JsonPropertyName("frameratedenominator")]
        public int FrameRateDenominator { get; set; }

        [JsonPropertyName("duration")]
        public double Duration { get; set; }

        public double FPS => (double)FrameRateNumerator / FrameRateDenominator;

        public int TotalFrames => (int)Math.Floor(FPS * Duration);
    }
}
