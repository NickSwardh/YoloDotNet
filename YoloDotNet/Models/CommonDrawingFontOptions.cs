namespace YoloDotNet.Models
{
    [EditorBrowsable(EditorBrowsableState.Never)]
    public class CommonDrawingFontOptions
    {
        /// <summary>
        /// Font used for drawing labels and confidence scores.
        /// </summary>
        public SKFont Font { get; set; } = ImageConfig.DefaultFont;

        /// <summary>
        /// Font used for drawing labels and confidence scores.
        /// </summary>
        public SKPaint FontColor { get; set; } = ImageConfig.FontColorPaint;

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
