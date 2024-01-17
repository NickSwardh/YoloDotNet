namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents the result of image classification
    /// </summary>
    public class Classification
    {
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
