// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    public class Segmentation : TrackingInfo, IDetection
    {
        /// <summary>
        /// Label information associated with the detected object.
        /// </summary>
        public LabelModel Label { get; init; } = new();

        /// <summary>
        /// Confidence score of the detected object.
        /// </summary>
        public double Confidence { get; init; }

        /// <summary>
        /// Rectangle defining the region of interest (bounding box) of the detected object.
        /// </summary>
        public SKRectI BoundingBox { get; init; }

        /// <summary>
        /// Bit-packed mask where each bit represents a pixel with confidence above a threshold (1 = present, 0 = absent).
        /// Can be unpacked to an <see cref="SKBitmap"/> using the <c>UnpackToBitmap</c> extension method.
        /// Pixel size is <see cref="PixelMaskSize"/>. When that size differs from <see cref="BoundingBox"/>, scale the mask onto the box.
        /// </summary>
        // Do not serialize the raw byte array directly, as it can be large and is not human-readable.
        [JsonIgnore] 
        public byte[] BitPackedPixelMask { get; set; } = [];

        /// <summary>
        /// Width of <see cref="BitPackedPixelMask"/> in pixels. 0 means the mask matches <see cref="BoundingBox"/> width.
        /// </summary>
        public int MaskWidth { get; init; }

        /// <summary>
        /// Height of <see cref="BitPackedPixelMask"/> in pixels. 0 means the mask matches <see cref="BoundingBox"/> height.
        /// </summary>
        public int MaskHeight { get; init; }

        /// <summary>
        /// Pixel size of <see cref="BitPackedPixelMask"/>. A zero <see cref="MaskWidth"/> or <see cref="MaskHeight"/> falls back to the bounding box.
        /// </summary>
        public (int Width, int Height) PixelMaskSize()
        {
            var width = MaskWidth > 0 ? MaskWidth : BoundingBox.Width;
            var height = MaskHeight > 0 ? MaskHeight : BoundingBox.Height;
            return (Math.Max(0, width), Math.Max(0, height));
        }
    }
}
