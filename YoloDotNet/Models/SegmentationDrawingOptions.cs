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

        /// <summary>
        /// Opacity level applied to the pixel mask.
        /// </summary>
        public int PixelMaskOpacity { get; set; } = 128;

        /// <summary>
        /// Whether to draw contour lines around the segmentation mask.
        /// </summary>
        public bool DrawContour { get; set; } = false;

        /// <summary>
        /// Thickness of the contour line in pixels.
        /// </summary>
        public int ContourThickness { get; set; } = 2;

        #region Mapping method
        public static explicit operator DetectionDrawingOptions(SegmentationDrawingOptions options) => new()
        {
            Font = options.Font,
            FontColor = options.FontColor,
            FontSize = options.FontSize,
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
            LabelPosition = options.LabelPosition,
        };
        #endregion
    }
}