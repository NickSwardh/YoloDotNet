// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

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
        /// Encoder to use when writing output.
        /// Only relevant if <see cref="VideoOutput"/> is set.
        /// </summary>
        public VideoEncoder VideoEncoder { get; set; }

        /// <summary>
        /// Gets or sets the width of the output video.
        /// Default is 0, which means the source width is preserved.
        /// Set to -2 to automatically calculate the width while maintaining the aspect ratio,
        /// based on the specified height.
        /// Note: Only one of Width or Height can be set to -2 at a time.
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// Gets or sets the height of the output video.
        /// Default is 0, which means the source height is preserved.
        /// Set to -2 to automatically calculate the height while maintaining the aspect ratio,
        /// based on the specified width.
        /// Note: Only one of Width or Height can be set to -2 at a time.
        /// </summary>
        public int Height { get; set; }

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

        /// <summary>
        /// Start time in seconds from which to begin processing the video.
        /// Only relevant if the input is a video file (not a live stream).
        /// </summary>
        public float StartTimeSeconds { get; set; }
        
        /// <summary>
        /// Duration in seconds for which to process the video.
        /// Only relevant if the input is a video file (not a live stream).
        /// </summary>
        public float DurationSeconds { get; set; }
     
    }
}
