namespace YoloDotNet.Models
{
    public class PoseEstimation : IDetection
    {
        /// <summary>
        /// Label information associated with the detected object.
        /// </summary>
        public LabelModel Label { get; init; } = new();

        /// <summary>
        /// Confidence score of the detected object.
        /// </summary>
        public double Confidence { get; init; }

        /// <summary>
        /// Rectangle defining the region of interest (bounding box) of the detected object.
        /// </summary>
        public Rectangle BoundingBox { get; init; }

        /// <summary>
        /// Pose estimations with x, y coordinates and confidence score
        /// </summary>
        public Pose[] PoseMarkers { get; set; } = [];
    }
}
