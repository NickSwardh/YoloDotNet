// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents options for configuring a Yolo object.
    /// </summary>
    public class YoloOptions
    {
        /// <summary>
        /// Gets or sets Execution Provider (CPU, CUDA or TensorRT).
        /// </summary>
        public IExecutionProvider ExecutionProvider { get; set; } = default!;

        /// <summary>
        /// Gets or sets the type of image resizing the onnx model requires.
        /// </summary>
        public ImageResize ImageResize { get; set; }

        /// <summary>
        /// SkiaSharp sampling options used when resizing the input image to the model input size.
        /// </summary>
        /// <remarks>
        /// - **Default:** Linear filtering (`SKFilterMode.Linear`) with no mipmap interpolation (`SKMipmapMode.None`).
        /// - **Accuracy:** Matches the bilinear resize YOLO models are trained and validated with.
        /// - **Performance:** `SKFilterMode.Nearest` is slightly faster, but changes the model input and therefore the detections.
        /// - **Modifiability:** This property can be changed at runtime to adjust filtering behavior.
        /// </remarks>
        public SKSamplingOptions SamplingOptions { get; set; } = ImageConfig.DefaultSamplingOptions;
    }
}
