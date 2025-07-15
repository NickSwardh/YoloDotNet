namespace YoloDotNet.Models
{
    public struct PoseEstimation
    {
        public PoseEstimation(Detection detection, KeyPoint[] keyPoints)
        {
            Detection = detection;
            KeyPoints = keyPoints;
        }

        public Detection Detection { get; }

        /// <summary>
        /// Keypoints with x, y coordinates and confidence score
        /// </summary>
        public KeyPoint[] KeyPoints { get; set; } = [];
    }
}
