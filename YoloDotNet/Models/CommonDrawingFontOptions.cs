// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    [EditorBrowsable(EditorBrowsableState.Never)]
    public class CommonDrawingFontOptions
    {
        /// <summary>
        /// Font used for drawing labels and confidence scores.
        /// </summary>
        public SKTypeface Font { get; set; } = SKTypeface.Default;

        /// <summary>
        /// Font size.
        /// </summary>
        public float FontSize { get; set; } = ImageConfig.FONT_SIZE;

        /// <summary>
        /// Font used for drawing labels and confidence scores.
        /// </summary>
        public SKColor FontColor { get; set; } = ImageConfig.FontColor;

        /// <summary>
        /// Whether to enable shadow effect behind text.
        /// </summary>
        public bool EnableFontShadow { get; set; } = true;

        /// <summary>
        /// Whether to dynamically scale font size and border thickness based on image resolution.
        /// </summary>
        public bool EnableDynamicScaling { get; set; } = true;

        /// <summary>
        /// Whether to draw label background.
        /// </summary>
        public bool DrawLabelBackground { get; set; } = true;

        /// <summary>
        /// Specifies the preferred position for labels relative to bounding boxes.
        /// Default is <see cref="LabelPosition.Auto"/>, which automatically determines the best position.
        /// </summary>
        public LabelPosition LabelPosition { get; set; } = LabelPosition.Auto;
    }
}
