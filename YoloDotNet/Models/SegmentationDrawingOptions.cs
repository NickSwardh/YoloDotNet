// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// Configuration options for drawing segmentation results.
    /// </summary>
    public class SegmentationDrawingOptions : CommonBoundingBoxOptions
    {
        /// <summary>
        /// Whether to draw pixelmask from segmentation result.
        /// </summary>
        public bool DrawSegmentationPixelMask { get; set; } = true;

        #region Mapping method
        public static explicit operator DetectionDrawingOptions(SegmentationDrawingOptions options) => new()
        {
            Font = options.Font,
            FontColor = options.FontColor,
            EnableFontShadow = options.EnableFontShadow,
            EnableDynamicScaling = options.EnableDynamicScaling,
            DrawLabelBackground = options.DrawLabelBackground,
            DrawBoundingBoxes = options.DrawBoundingBoxes,
            DrawLabels = options.DrawLabels,
            DrawConfidenceScore = options.DrawConfidenceScore,
            DrawTrackedTail = options.DrawTrackedTail,
            BorderThickness = options.BorderThickness,
            BoundingBoxOpacity = options.BoundingBoxOpacity,
            BoundingBoxHexColors = options.BoundingBoxHexColors,
            TailThickness = options.TailThickness,
            TailPaintColorStart = options.TailPaintColorStart,
            TailPaintColorEnd = options.TailPaintColorEnd,
        };
        #endregion
    }
}