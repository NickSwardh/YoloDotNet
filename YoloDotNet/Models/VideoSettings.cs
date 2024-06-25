namespace YoloDotNet.Models
{
    /// <summary>
    /// Extracted metadata from video
    /// </summary>
    public class VideoSettings
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
        public double? FPS { get; set; }

        /// <summary>
        /// Total number of frames in the video.
        /// </summary>
        public int TotalFrames { get; set; }

        /// <summary>
        /// Width of the video.
        /// </summary>
        public int? Width { get; set; }

        /// <summary>
        /// Height of the video.
        /// </summary>
        public int? Height { get; set; }

        /// <summary>
        /// Compression Quality 0-51
        /// </summary>
        public int Quality { get; set; }

        /// <summary>
        /// Output folder for processed video file.
        /// </summary>
        public string OutputFolder { get; set; } = default!;

        /// <summary>
        /// Temporary folder used during video processing.
        /// </summary>
        public string TempFolder { get; set; } = default!;

        /// <summary>
        /// Keep processed frames in temp folder
        /// </summary>
        public bool KeepFrames { get; set; }

        /// <summary>
        /// Keep audio in video output
        /// </summary>
        public bool KeepAudio { get; set; }

        /// <summary>
        /// Draw labels on image
        /// </summary>
        public bool GenerateVideo { get; set; }

        /// <summary>
        /// Draw labels on processed frames
        /// </summary>
        public bool DrawLabels { get; set; }

        /// <summary>
        /// Draw confidence percentage on frames
        /// </summary>
        public bool DrawConfidence { get; set; }

        /// <summary>
        /// Draw boundingbox and/or pixelmask for segmentation
        /// </summary>
        public DrawSegment DrawSegment { get; set; }

        /// <summary>
        /// Options for drawing keypoints.
        /// </summary>
        public KeyPointOptions KeyPointOptions { get; set; } = new();

        #region Mapping methods

        /// <summary>
        /// Map VideoOptions to VideoMetaData
        /// </summary>
        /// <param name="result"></param>
        public static explicit operator VideoSettings(VideoOptions result) => new()
        {
            VideoFile = result.VideoFile,
            FPS = result.FPS,
            Width = result.Width,
            Height = result.Height,
            Quality = result.Quality,
            OutputFolder = result.OutputDir,
            TempFolder = Path.Combine(result.OutputDir, nameof(FolderName.Temp)),
            KeepFrames = result.KeepFrames,
            KeepAudio = result.KeepAudio,
            GenerateVideo = result.GenerateVideo,
            DrawLabels = result.DrawLabels,
            DrawConfidence = result.KeepAudio,
            DrawSegment = result.DrawSegment,
            KeyPointOptions = result.KeyPointOptions,
        };

        #endregion

    }
}
