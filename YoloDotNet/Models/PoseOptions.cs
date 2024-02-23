namespace YoloDotNet.Models
{
    public class PoseOptions
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
        public string DefaultPoseColor { get; init; } = "#FF69B4"; // Hotpink

        /// <summary>
        /// User-defined mapping to determine how to connect pose-markers and specify associated colors.
        /// </summary>
        public PoseMap[] PoseMappings { get; init; } = [];
    }

    /// <summary>
    /// 
    /// </summary>
    public class PoseMap
    {
        public string Color { get; init; } = default!;
        public PoseConnection[] Connections { get; init; } = [];
    }

    /// <summary>
    /// Represents a connection between pose-markers and their parent marker, defined by index and color.
    /// </summary>
    /// <param name="Index"></param>
    /// <param name="Color"></param>
    public record PoseConnection(int Index, string Color);
}
