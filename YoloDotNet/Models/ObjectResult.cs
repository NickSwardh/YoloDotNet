// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    public class ObjectResult
    {
        /// <summary>
        /// Label information associated with the detected object.
        /// </summary>
        public LabelModel Label { get; set; } = new();

        /// <summary>
        /// Confidence score of the detected object.
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// Region of interest (bounding box) of the detected object.
        /// </summary>
        public SKRectI BoundingBox { get; set; }

        /// <summary>
        /// Region of interest (bounding box) of the detected object for ONNX model dimensions.
        /// </summary>
        public SKRect BoundingBoxUnscaled { get; set; }

        /// <summary>
        /// Index of bounding box
        /// </summary>
        public int BoundingBoxIndex { get; set; }

        /// <summary>
        /// Bit-packed mask where each bit represents a pixel with confidence above a threshold (1 = present, 0 = absent).
        /// </summary>
        public byte[] BitPackedPixelMask { get; set; } = [];

        /// <summary>
        /// Confidence value, X and Y coordinates for Pose Estimation key points
        /// </summary>
        public KeyPoint[] KeyPoints { get; set; } = [];

        /// <summary>
        /// Orientation angle of the bounding box for OBB detections.
        /// </summary>
        public float OrientationAngle { get; set; }

        #region Mapping methods
        public static explicit operator ObjectDetection(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox
        };

        public static explicit operator OBBDetection(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox,
            OrientationAngle = result.OrientationAngle
        };

        public static explicit operator Segmentation(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox,
            BitPackedPixelMask = result.BitPackedPixelMask
        };

        public static explicit operator PoseEstimation(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox,
            KeyPoints = result.KeyPoints
        };
        #endregion
    }
}
