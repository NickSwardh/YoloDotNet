namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents the result of object detection, including label information, confidence score, and bounding box.
    /// </summary>
    public struct ObjectDetection
    {
        public ObjectDetection(Detection detection)
        {
            Detection = detection;
        }

        public Detection Detection { get; }

    }
}
