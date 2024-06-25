namespace YoloDotNet.Models
{
    public class ObjectResult
    {
        /// <summary>
        /// Label information associated with the detected object.
        /// </summary>
        public LabelModel Label { get; set; } = new();

        /// <summary>
        /// Confidence score of the detected object.
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// Region of interest (bounding box) of the detected object.
        /// </summary>
        public Rectangle BoundingBox { get; set; }

        /// <summary>
        /// Index of bounding box
        /// </summary>
        public int BoundingBoxIndex { get; set; }

        /// <summary>
        /// Segmented pixels coordinates (x,y) and pixel confidence value
        /// </summary>
        public Pixel[] SegmentedPixels { get; set; } = [];

        public KeyPoint[] KeyPoints { get; set; } = [];

        /// <summary>
        /// Orientation angle of the bounding box for OBB detections.
        /// </summary>
        public float OrientationAngle { get; set; }

        #region Mapping methods
        public static explicit operator ObjectDetection(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox
        };

        public static explicit operator OBBDetection(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox,
            OrientationAngle = result.OrientationAngle
        };

        public static explicit operator Segmentation(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox,
            SegmentedPixels = result.SegmentedPixels
        };

        public static explicit operator PoseEstimation(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox,
            KeyPoints = result.KeyPoints
        };
        #endregion
    }
}
