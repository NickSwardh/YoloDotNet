﻿namespace YoloDotNet.Models
{
    public class PoseEstimation : TrackingInfo, IDetection
    {
        /// <summary>
        /// Label information associated with the detected object.
        /// </summary>
        public LabelModel Label { get; init; } = new();

        /// <summary>
        /// Confidence score of the detected object.
        /// </summary>
        public double Confidence { get; init; }

        /// <summary>
        /// Rectangle defining the region of interest (bounding box) of the detected object.
        /// </summary>
        public SKRectI BoundingBox { get; init; }

        /// <summary>
        /// Keypoints with x, y coordinates and confidence score
        /// </summary>
        public KeyPoint[] KeyPoints { get; set; } = [];
    }
}
