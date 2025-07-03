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
    }
}
