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
        /// Gets or sets the file path to the ONNX model.
        /// </summary>
        public string OnnxModel { get; set; } = default!;

        /// <summary>
        /// Gets or sets the ONNX model as a byte array.
        /// </summary>
        public byte[]? OnnxModelBytes { get; set; }

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
