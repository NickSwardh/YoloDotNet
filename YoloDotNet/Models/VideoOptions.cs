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
        /// Set width in of output video.
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// Set height of output video.
        /// </summary>
        public int Height { get; set; }

        /// <summary>
        /// Set frames per second (FPS) of output video.
        /// </summary>
        public double FPS { get; set; }

        /// <summary>
        /// Compression Quality (0-51) 28 = Default.
        /// </summary>
        public int Quality { get; set; } = 30;


        /// <summary>
        /// Generate a new video from processed video frames.
        /// </summary>
        public bool GenerateVideo { get; set; } = true;
    }
}
