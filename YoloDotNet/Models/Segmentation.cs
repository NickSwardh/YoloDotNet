using SixLabors.ImageSharp;

namespace YoloDotNet.Models
{
    public class Segmentation : IDetection
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
        public Rectangle Rectangle { get; init; }

        /// <summary>
        /// Segmentated pixels (x,y) with the pixel confidence value
        /// </summary>
        public double[,] SegmentationMask { get; set; } = default!;
    }
}
