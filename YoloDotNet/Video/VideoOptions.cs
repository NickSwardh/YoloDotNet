namespace YoloDotNet.Video
{
    /// <summary>
    /// Options for configuring video processing.
    /// </summary>
    public class VideoOptions
    {
        /// <summary>
        /// Path to the video file or the URL of a live video stream to process.
        /// </summary>
        public string VideoInput { get; set; } = default!;

        /// <summary>
        /// Optional: Path to the output video file where the processed video will be saved.
        /// </summary>
        public string VideoOutput{ get; set; } = default!;

        /// <summary>
        /// Gets or sets the width of the output video. The default is 1080. 
        /// Set to -2 to automatically calculate the width while maintaining the aspect ratio.
        /// Note: Only one of width or height can be set to -2 at a time.
        /// </summary>
        public int Width { get; set; } = 1080;

        /// <summary>
        /// Gets or sets the height of the output video. The default is -2, meaning the height will be automatically 
        /// calculated to maintain the aspect ratio based on the width.
        /// Note: Only one of width or height can be set to -2 at a time.
        /// </summary>
        public int Height { get; set; } = -2;

        /// <summary>
        /// Set frames per second (FPS) of output video.
        /// </summary>
        public FrameRate FrameRate { get; set; } = FrameRate.AUTO;

        /// <summary>
        /// Gets or sets the interval at which frames are processed. 
        /// Only every nth frame will be processed, where n is the value of this property.
        /// Useful for surveillance or monitoring scenarios where processing every frame is unnecessary.
        /// </summary>
        public int FrameInterval { get; set; }

        /// <summary>
        /// Duration of each video segment in seconds. The video will be split into chunks of this length.
        /// Default is 600 (10 minutes).
        /// </summary>
        public int VideoChunkDuration { get; set; } = 600;

        /// <summary>
        /// Compression quality for the output video (0–51). Lower is better quality. Default is 30.
        /// </summary>
        public int CompressionQuality { get; set; } = 30;
    }
}
