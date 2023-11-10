namespace YoloDotNet.Models
{
    /// <summary>
    /// Extracted metadata from video
    /// </summary>
    public class VideoMetaData
    {
        /// <summary>
        /// Path to the video file.
        /// </summary>
        public string VideoFile { get; set; } = default!;

        /// <summary>
        /// Duration of the video.
        /// </summary>
        public TimeSpan Duration { get; set; }

        /// <summary>
        /// Frames per second (FPS).
        /// </summary>
        public double FPS { get; set; }

        /// <summary>
        /// Total number of frames in the video.
        /// </summary>
        public int Frames { get; set; }

        /// <summary>
        /// Width of the video.
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// Height of the video.
        /// </summary>
        public int Height { get; set; }

        /// <summary>
        /// Output folder for processed video file.
        /// </summary>
        public DirectoryInfo OutputFolder { get; set; } = default!;

        /// <summary>
        /// Temporary folder used during video processing.
        /// </summary>
        public DirectoryInfo TempFolder { get; set; } = default!;
    }
}
