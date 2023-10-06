namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents the configuration and metadata of an ONNX model for object detection.
    /// </summary>
    public record OnnxModel
    {
        /// <summary>
        /// Name of the input tensor in the ONNX model.
        /// </summary>
        public string InputName { get; init; } = default!;

        /// <summary>
        /// Name of the output tensor in the ONNX model.
        /// </summary>
        public string OutputName { get; init; } = default!;

        /// <summary>
        /// Creation date of the ONNX model.
        /// </summary>
        public DateTime Date { get; init; }

        /// <summary>
        /// Description of the ONNX model.
        /// </summary>
        public string Description { get; init; } = default!;

        /// <summary>
        /// Author of the ONNX model.
        /// </summary>
        public string Author { get; init; } = default!;

        /// <summary>
        /// Task or purpose of the ONNX model.
        /// </summary>
        public string Task { get; init; } = default!;

        /// <summary>
        /// License information associated with the ONNX model.
        /// </summary>
        public string License { get; init; } = default!;

        /// <summary>
        /// Version of the ONNX model.
        /// </summary>
        public string Version { get; init; } = default!;

        /// <summary>
        /// Stride value used in object detection.
        /// </summary>
        public int Stride { get; init; }

        /// <summary>
        /// Batch size used in object detection.
        /// </summary>
        public int BatchSize { get; init; }

        /// <summary>
        /// Size of input images expected by the ONNX model.
        /// </summary>
        public Size ImageSize { get; init; }

        /// <summary>
        /// Input tensor configuration of the ONNX model.
        /// </summary>
        public Input Input { get; init; } = default!;

        /// <summary>
        /// Output tensor configuration of the ONNX model.
        /// </summary>
        public Output Output { get; init; } = default!;

        /// <summary>
        /// Array of label models for object detection.
        /// </summary>
        public LabelModel[] Labels { get; init; } = default!;
    }

    /// <summary>
    /// Represents the configuration of input data for the ONNX model.
    /// </summary>
    /// <param name="BatchSize">The batch size of input data.</param>
    /// <param name="Channels">The number of input channels.</param>
    /// <param name="Width">The width of input data.</param>
    /// <param name="Height">The height of input data.</param>
    public record Input(int BatchSize, int Channels, int Width, int Height);

    /// <summary>
    /// Represents the configuration of output data from the ONNX model.
    /// </summary>
    /// <param name="BatchSize">The batch size of output data.</param>
    /// <param name="Dimensions">The number of dimensions in the output data.</param>
    /// <param name="Channels">The number of output channels.</param>
    public record Output(int BatchSize, int Dimensions, int Channels);
}
