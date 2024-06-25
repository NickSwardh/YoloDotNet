namespace YoloDotNet.Models
{
    /// <summary>
    /// Options for configuring video processing.
    /// </summary>
    public class VideoOptions
    {
        /// <summary>
        /// Path to the input video file.
        /// </summary>
        public string VideoFile { get; set; } = default!;

        /// <summary>
        /// Output directory for the processed video file.
        /// </summary>
        public string OutputDir { get; set; } = default!;

        /// <summary>
        /// Keep frames after processing.
        /// </summary>
        public bool KeepFrames { get; set; } = false;

        /// <summary>
        /// Keep audio track in processed video.
        /// </summary>
        public bool KeepAudio { get; set; } = true;

        /// <summary>
        /// Generate a new video from processed video frames.
        /// </summary>
        public bool GenerateVideo { get; set; } = true;

        /// <summary>
        /// Draw labels on processed frames
        /// </summary>
        public bool DrawLabels { get; set; } = true;

        /// <summary>
        /// Draw confidence scores on labels.
        /// </summary>
        public bool DrawConfidence { get; set; } = true;

        /// <summary>
        /// Set frames per second (FPS) of output video.
        /// </summary>
        public float? FPS { get; set; }

        /// <summary>
        /// Set width in of output video.
        /// </summary>
        public int? Width { get; set; }

        /// <summary>
        /// Set height of output video.
        /// </summary>
        public int? Height { get; set; }

        /// <summary>
        /// Compression Quality (0-51) 28 = Default.
        /// </summary>
        public int Quality { get; set; } = 28;

        /// <summary>
        /// Draw boundingboxes and/or pixelmasks on segmented objects.
        /// </summary>
        public DrawSegment DrawSegment { get; set; }

        /// <summary>
        /// Options for drawing pose estimation markers.
        /// </summary>
        public KeyPointOptions KeyPointOptions { get; set; } = new();
    }
}
