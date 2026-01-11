// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    [EditorBrowsable(EditorBrowsableState.Never)]
    public class CommonBoundingBoxOptions : CommonDrawingFontOptions
    {
        /// <summary>
        /// Whether to draw bounding boxes around detected objects.
        /// </summary>
        public bool DrawBoundingBoxes { get; set; } = true;

        /// <summary>
        /// Whether to draw labels on detected objects.
        /// </summary>
        public bool DrawLabels { get; set; } = true;

        /// <summary>
        /// Whether to draw confidence scores next to labels.
        /// </summary>
        public bool DrawConfidenceScore { get; set; } = true;

        /// <summary>
        /// Whether to draw the tail of tracked objects.
        /// </summary>
        public bool DrawTrackedTail { get; set; } = true;

        /// <summary>
        /// Thickness of bounding box borders, in pixels.
        /// </summary>
        public float BorderThickness { get; set; } = ImageConfig.BORDER_THICKNESS;

        /// <summary>
        /// Opacity level for bounding boxes (0-255).
        /// </summary>
        public int BoundingBoxOpacity { get; set; } = ImageConfig.DEFAULT_OPACITY;

        /// <summary>
        /// Array of bounding box colors as hex strings (e.g., "#FF0000").
        /// If no colors are specified, default colors will be used.
        /// </summary>
        public string[] BoundingBoxHexColors { get; set; } = YoloDotNetColors.Get();

        /// <summary>
        /// Thickness of the tail line drawn for object tracking visualization.
        /// </summary>
        public float TailThickness { get; set; } = ImageConfig.TailThickness;

        /// <summary>
        /// Starting color of the gradient used in the tracked tail line.
        /// </summary>
        public SKColor TailPaintColorStart { get; set; } = ImageConfig.TailPaintColorStart;

        /// <summary>
        /// Ending color of the gradient used in the tracked tail line.
        /// </summary>
        public SKColor TailPaintColorEnd { get; set; } = ImageConfig.TailPaintColorEnd;
    }
}
