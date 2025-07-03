namespace YoloDotNet.Models
{
    public class PoseDrawingOptions : CommonBoundingBoxOptions
    {
        /// <summary>
        /// Confidence threshold for displaying keypoints.
        /// </summary>
        public double PoseConfidence { get; set; } = ImageConfig.POSE_KEYPOINT_THRESHOLD;

        /// <summary>
        /// Default color used to draw keypoints when no specific colors or markers are provided via <see cref="KeyPointMarkers"/>.
        /// </summary>
        public SKColor DefaultPoseColor { get; set; } = ImageConfig.PoseMarkerColor;

        /// <summary>
        /// User-defined mapping to determine how to connect keypoints and specify associated colors.
        /// </summary>
        public KeyPointMarker[] KeyPointMarkers { get; set; } = [];

        #region Mapping method
        public static explicit operator DetectionDrawingOptions(PoseDrawingOptions options) => new()
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

    /// <summary>
    /// Represents a mapping between keypoints and their connections.
    /// </summary>
    public record KeyPointMarker
    {
        /// <summary>
        /// Color associated with the keypoint.
        /// </summary>
        public string Color { get; init; } = default!;

        /// <summary>
        /// Defines connections between keypoints.
        /// </summary>
        public KeyPointConnection[] Connections { get; init; } = [];
    }

    /// <summary>
    /// Represents a connection between pose-markers and their parent marker, defined by index and color.
    /// </summary>
    /// <param name="Index"></param>
    /// <param name="Color"></param>
    public record KeyPointConnection(int Index, string Color);
}
