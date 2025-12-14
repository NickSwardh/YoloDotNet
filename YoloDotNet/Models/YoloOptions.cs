// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
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
        /// SkiaSharp sampling options optimized for efficient downscaling.
        /// </summary>
        /// <remarks>
        /// - **Default:** Linear filtering (`SKFilterMode.Linear`) with no mipmap interpolation (`SKMipmapMode.None`).
        /// - **Performance:** Fast and efficient for downscaling.
        /// - **Quality:** Produces smooth results with minimal aliasing.
        /// - **Best Use Case:** Ideal when reducing image size while maintaining a balance between speed and quality.
        /// - **Modifiability:** This property can be changed at runtime to adjust filtering behavior.
        /// </remarks>
        public SKSamplingOptions SamplingOptions { get; set; } = ImageConfig.DefaultSamplingOptions;
    }
}
