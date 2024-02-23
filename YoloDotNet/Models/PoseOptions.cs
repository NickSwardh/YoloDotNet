namespace YoloDotNet.Models
{
    public record PoseOptions
    {
        /// <summary>
        /// Confidence threshold for displaying markers.
        /// </summary>
        public double PoseConfidence { get; init; } = 0.65;

        /// <summary>
        /// Draw bounding box on detected object.
        /// </summary>
        public bool DrawBoundingBox { get; init; } = true;

        /// <summary>
        /// Default color for pos-markers.
        /// </summary>
        public string DefaultPoseColor { get; init; } = "#FFF633"; // Yellow

        /// <summary>
        /// User-defined mapping to determine how to connect pose-markers and specify associated colors.
        /// </summary>
        public PoseMarker[] PoseMarkers { get; init; } = [];
    }

    /// <summary>
    /// Represents a mapping between pose markers and their connections.
    /// </summary>
    public record PoseMarker
    {
        /// <summary>
        /// Color associated with the pose markers.
        /// </summary>
        public string Color { get; init; } = default!;

        /// <summary>
        /// Defines connections between pose markers and their parent marker.
        /// </summary>
        public PoseConnection[] Connections { get; init; } = [];
    }

    /// <summary>
    /// Represents a connection between pose-markers and their parent marker, defined by index and color.
    /// </summary>
    /// <param name="Index"></param>
    /// <param name="Color"></param>
    public record PoseConnection(int Index, string Color);
}
