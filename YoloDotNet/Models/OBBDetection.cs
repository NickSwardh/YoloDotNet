namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents the result of object detection, including label information, confidence score, and bounding box.
    /// </summary>
    public struct OBBDetection
    {
        public OBBDetection(Detection detection, float orientationAngle)
        {
            Detection = detection;
            OrientationAngle = orientationAngle;
        }

        public Detection Detection { get; }

        /// <summary>
        /// Orientation angle of bounding box
        /// </summary>
        public float OrientationAngle { get; set; }
    }
}
