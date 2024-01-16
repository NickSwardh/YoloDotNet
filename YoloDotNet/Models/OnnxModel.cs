using SixLabors.ImageSharp;
using YoloDotNet.Enums;

namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents the configuration and metadata of an ONNX model for object detection.
    /// </summary>
    public record OnnxModel
    {
        public ModelType ModelType { get; init; }

        /// <summary>
        /// Name of the input tensor in the ONNX model.
        /// </summary>
        public string InputName { get; init; } = default!;

        /// <summary>
        /// Name of the output tensor in the ONNX model.
        /// </summary>
        public string OutputName { get; init; } = default!;

        public Dictionary<string, string> CustomMetaData { get; set; } = new();

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
        public IOutputShape Output { get; init; } = default!;

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
    /// <param name="Elements">The number of dimensions in the output data.</param>
    public record ClassificationShape(int BatchSize, int Elements) : IOutputShape;

    /// <summary>
    /// Represents the configuration of output data from the ONNX model.
    /// </summary>
    /// <param name="BatchSize">The batch size of output data.</param>
    /// <param name="Elements">The number of dimensions in the output data.</param>
    /// <param name="Channels">The number of output channels.</param>
    public record ObjectDetectionShape(int BatchSize, int Elements, int Channels) : IOutputShape;
}