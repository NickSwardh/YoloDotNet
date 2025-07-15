namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents the result of image classification
    /// </summary>
    public struct Classification
    {
        public Classification(string label, double confidence)
        {
            this.Label = label;
            this.Confidence = confidence;
        }

        /// <summary>
        /// Label of classified image.
        /// </summary>
        public string Label { get; set; } = default!;

        /// <summary>
        /// Confidence score of classified image.
        /// </summary>
        public double Confidence { get; set; }
    }
}
