﻿// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents the result of object detection, including label information, confidence score, and bounding box.
    /// </summary>
    public class OBBDetection : TrackingInfo, IDetection
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
        /// Orientation angle of bounding box
        /// </summary>
        public float OrientationAngle { get; set; }

    }
}
