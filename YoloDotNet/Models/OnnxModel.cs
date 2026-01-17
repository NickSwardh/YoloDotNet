// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents the configuration and metadata of an ONNX model for object detection.
    /// </summary>
    public record OnnxModel
    {
        /// <summary>
        /// Gets the type of the model, eg. Object Detection, Classification etc.
        /// </summary>
        public ModelType ModelType { get; init; }

        /// <summary>
        /// Gets or the version of the model being used.
        /// </summary>
        public ModelVersion ModelVersion { get; set; }

        /// <summary>
        /// Gets the data type of the model. Float32 or Float16.
        /// </summary>
        public ModelDataType ModelDataType { get; init; }

        public Dictionary<string, long[]> InputShapes { get; init; } = default!;
        public Dictionary<string, int[]> OutputShapes { get; init; } = default!;

        /// <summary>
        /// Array of label models for object detection.
        /// </summary>
        public LabelModel[] Labels { get; init; } = default!;

        /// <summary>
        /// Gets the size of the input shape used for tensor allocation and array pooling.
        /// </summary>
        public int InputShapeSize { get; init; }

        /// <summary>
        /// ONNX custom metadata 
        /// </summary>
        public Dictionary<string, string> CustomMetaData { get; set; } = [];
    }

    ///// <summary>
    ///// Represents the configuration of input data for the ONNX model in BCHW oder
    ///// [Batch, Channels, Height, Width]
    ///// </summary>
    ///// <param name="BatchSize">The batch size of input data.</param>
    ///// <param name="Channels">The number of input channels.</param>
    ///// <param name="Height">The height of input data.</param>
    ///// <param name="Width">The width of input data.</param>
    //public record Input(int BatchSize, int Channels, int Height, int Width)
    //{
    //    public static Input Shape(int[] dimensions)
    //        => new(dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    //}

    ///// <summary>
    ///// Represents the configuration of output data for the ONNX model.
    ///// </summary>
    ///// <param name="BatchSize">The batch size of input data.</param>
    ///// <param name="Elements">The number of elements of input data.</param>
    ///// <param name="Channels">The number of channels of input data.</param>
    ///// <param name="Width">The width of input data.</param>
    ///// <param name="Height">The height of input data.</param>
    //public record Output(int BatchSize, int Elements, int Channels, int Width, int Height)
    //{
    //    public static Output Classification(int[] dimensions)
    //        => new(dimensions[0], dimensions[1], 0, 0, 0);

    //    public static Output Detection(int[] dimensions)
    //        => new(dimensions[0], dimensions[1], dimensions[2], 0, 0);

    //    public static Output Segmentation(int[] dimensions)
    //        => new(dimensions[0], 0, dimensions[1], dimensions[2], dimensions[3]);

    //    public static Output Empty() => new(0, 0, 0, 0, 0);
    //}
}