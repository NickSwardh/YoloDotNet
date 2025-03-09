namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents options for configuring a Yolo object.
    /// </summary>
    public class YoloOptions
    {
        /// <summary>
        /// Gets or sets the file path to the ONNX model.
        /// </summary>
        public string OnnxModel { get; set; } = default!;

        /// <summary>
        /// Gets or sets the type of the model (e.g., detection, classification).
        /// </summary>
        public ModelType ModelType { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to use CUDA for GPU acceleration (default is true).
        /// </summary>
        public bool Cuda { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to prime the GPU before inference (default is false).
        /// </summary>
        public bool PrimeGpu { get; set; } = false;

        /// <summary>
        /// Gets or sets the ID of the GPU to use (default is 0).
        /// </summary>
        public int GpuId { get; set; } = 0;

        /// <summary>
        /// Get or sets the type of image resizing the onnx model requires.
        /// </summary>
        public ImageResize ImageResize { get; set; }
    }
}
