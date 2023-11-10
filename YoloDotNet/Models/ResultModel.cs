namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents the result of object detection, including label information, confidence score, and bounding box.
    /// </summary>
    public record ResultModel
    {
        /// <summary>
        /// Label information associated with the detected object.
        /// </summary>
        public LabelModel Label { get; init; } = default!;

        /// <summary>
        /// Confidence score of the detected object.
        /// </summary>
        public double Confidence { get; init; }

        /// <summary>
        /// Rectangle defining the region of interest (bounding box) of the detected object.
        /// </summary>
        public RectangleF Rectangle { get; init; }
    }
}
