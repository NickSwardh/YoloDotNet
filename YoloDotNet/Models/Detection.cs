namespace YoloDotNet.Models
{
    public struct Detection
    {
        public Detection(LabelModel label, double confidence, SKRectI boundingBox)
        {
            Label = label;
            Confidence = confidence;
            BoundingBox = boundingBox;
        }

        /// <summary>
        /// Label information associated with the detected object.
        /// </summary>
        public LabelModel Label { get; }

        /// <summary>
        /// Confidence score of the detected object.
        /// </summary>
        public double Confidence { get; }

        /// <summary>
        /// Rectangle defining the region of interest (bounding box) of the detected object.
        /// </summary>
        public SKRectI BoundingBox { get; }
    }
}
