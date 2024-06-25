namespace YoloDotNet.Models
{
    public record KeyPointOptions
    {
        /// <summary>
        /// Confidence threshold for displaying keypoints.
        /// </summary>
        public double PoseConfidence { get; init; } = 0.65;

        /// <summary>
        /// Draw bounding box on detected object.
        /// </summary>
        public bool DrawBoundingBox { get; init; } = true;

        /// <summary>
        /// Keypoint Default color.
        /// </summary>
        public string DefaultPoseColor { get; init; } = "#FFF633"; // Yellow

        /// <summary>
        /// User-defined mapping to determine how to connect keypoints and specify associated colors.
        /// </summary>
        public KeyPointMarker[] PoseMarkers { get; init; } = [];
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
