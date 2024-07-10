namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents the configuration and metadata of an ONNX model for object detection.
    /// </summary>
    public record OnnxModel
    {
        public ModelType ModelType { get; init; }

        public ModelVersion ModelVersion { get; set; }

        /// <summary>
        /// Name of the input tensor in the ONNX model.
        /// </summary>
        public string InputName { get; init; } = default!;

        /// <summary>
        /// Name of the output tensors in the ONNX model.
        /// </summary>
        public List<string> OutputNames { get; init; } = default!;

        /// <summary>
        /// Input tensor configuration of the ONNX model.
        /// </summary>
        public Input Input { get; init; } = default!;

        /// <summary>
        /// Output tensor configuration of the ONNX model.
        /// </summary>
        public List<Output> Outputs { get; init; } = default!;

        /// <summary>
        /// Array of label models for object detection.
        /// </summary>
        public LabelModel[] Labels { get; init; } = default!;

        /// <summary>
        /// The ONNX-model input shape for creating a Tensor
        /// </summary>
        public long[] InputShape { get; init; } = default!;

        /// <summary>
        /// ONNX custom metadata 
        /// </summary>
        public Dictionary<string, string> CustomMetaData { get; set; } = [];
    }

    /// <summary>
    /// Represents the configuration of input data for the ONNX model.
    /// </summary>
    /// <param name="BatchSize">The batch size of input data.</param>
    /// <param name="Channels">The number of input channels.</param>
    /// <param name="Width">The width of input data.</param>
    /// <param name="Height">The height of input data.</param>
    public record Input(int BatchSize, int Channels, int Width, int Height)
    {
        public static Input Shape(int[] dimensions)
            => new(dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    }

    /// <summary>
    /// Represents the configuration of output data for the ONNX model.
    /// </summary>
    /// <param name="BatchSize">The batch size of input data.</param>
    /// <param name="Elements">The number of elements of input data.</param>
    /// <param name="Channels">The number of channels of input data.</param>
    /// <param name="Width">The width of input data.</param>
    /// <param name="Height">The height of input data.</param>
    public record Output(int BatchSize, int Elements, int Channels, int Width, int Height)
    {
        public static Output Classification(int[] dimensions)
            => new(dimensions[0], dimensions[1], 0, 0, 0);

        public static Output Detection(int[] dimensions)
            => new(dimensions[0], dimensions[1], dimensions[2], 0, 0);

        public static Output Segmentation(int[] dimensions)
            => new(dimensions[0], 0, dimensions[1], dimensions[2], dimensions[3]);

        public static Output Empty() => new(0, 0, 0, 0, 0);
    }
}