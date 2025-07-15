namespace YoloDotNet.Models
{
    public struct Segmentation
    {
        public Segmentation(Detection detection, Pixel[] segmentedPixels)
        {
            Detection = detection;
            SegmentedPixels = segmentedPixels;
        }

        public Detection Detection { get; }

        /// <summary>
        /// Segmentated pixels (x,y) with the pixel confidence value
        /// </summary>
        public Pixel[] SegmentedPixels { get; set; } = [];
    }
}
