// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    public class Segmentation : TrackingInfo, IDetection
    {
        private byte[]? _bitPackedPixelMask;

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
        /// Outlines of the segmentation mask: closed polygons in image coordinates, with sub-pixel precision, including
        /// holes and separate parts, cropped to the <see cref="BoundingBox"/>. Filling them with the even-odd rule gives
        /// the mask; a pixel belongs to it when its centre lies inside.
        /// </summary>
        /// <remarks>
        /// This is the stored form of the mask. Use <c>GetPixelMask</c> for a mask bitmap (optionally anti-aliased) or
        /// <see cref="BitPackedPixelMask"/> for a compact pixel mask.
        /// </remarks>
        // Serialized separately by the result exporters.
        [JsonIgnore]
        public SKPoint[][] Contours { get; init; } = [];

        /// <summary>
        /// Bit-packed mask covering the <see cref="BoundingBox"/>, row by row, where each bit represents a pixel
        /// (1 = part of the object, 0 = absent), least significant bit first.
        /// For a mask bitmap, use the <c>GetPixelMask</c> extension method.
        /// </summary>
        /// <remarks>
        /// Rasterized from <see cref="Contours"/> the first time it is read, then kept.
        /// </remarks>
        // Do not serialize the raw byte array directly, as it can be large and is not human-readable.
        [JsonIgnore]
        public byte[] BitPackedPixelMask
        {
            get => _bitPackedPixelMask ?? LazyInitializer.EnsureInitialized(ref _bitPackedPixelMask, () => MaskRasterizer.ToBitPacked(Contours, BoundingBox));
            set => _bitPackedPixelMask = value;
        }
    }
}
