// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Utils
{
    /// <summary>
    /// Builds a segmentation pixel mask by combining the model's mask prototypes with a set of
    /// per-detection mask weights.
    ///
    /// The prototype tensor is laid out as [channels, height, width]. A naive implementation that
    /// loops pixel-outer / channel-inner strides through memory by an entire prototype plane
    /// (width * height) on every channel step, which is very cache-unfriendly. This implementation
    /// instead loops channel-outer / pixel-inner so each prototype plane is read sequentially, and
    /// vectorizes the inner accumulation with FMA when available.
    /// </summary>
    internal static unsafe class SegmentationMaskProcessor
    {
        /// <summary>
        /// Accumulates the weighted prototype sum for the given mask region, applies the sigmoid
        /// activation and writes the result into the supplied Gray8 bitmap.
        /// </summary>
        /// <param name="bitmap">Destination Gray8 bitmap (mask resolution).</param>
        /// <param name="prototypes">Mask prototype tensor, laid out as [channels, height, width].</param>
        /// <param name="weights">Per-detection mask weights (one per prototype channel).</param>
        /// <param name="channels">Number of prototype channels to combine.</param>
        /// <param name="maskWidth">Prototype/mask width.</param>
        /// <param name="maskHeight">Prototype/mask height.</param>
        /// <param name="startX">Inclusive left bound of the region within the mask.</param>
        /// <param name="startY">Inclusive top bound of the region within the mask.</param>
        /// <param name="endX">Inclusive right bound of the region within the mask.</param>
        /// <param name="endY">Inclusive bottom bound of the region within the mask.</param>
        public static void Apply(
            SKBitmap bitmap,
            ReadOnlySpan<float> prototypes,
            ReadOnlySpan<float> weights,
            int channels,
            int maskWidth,
            int maskHeight,
            int startX,
            int startY,
            int endX,
            int endY)
        {
            if (endX < startX || endY < startY)
                return;

            int regionWidth = endX - startX + 1;
            int regionHeight = endY - startY + 1;
            int regionSize = regionWidth * regionHeight;
            int planeSize = maskWidth * maskHeight;

            // Per-region accumulator. Rented to avoid per-detection heap allocations.
            var accBuffer = ArrayPool<float>.Shared.Rent(regionSize);

            try
            {
                Array.Clear(accBuffer, 0, regionSize);

                fixed (float* pProto = prototypes)
                fixed (float* pAcc = accBuffer)
                {
                    // Channel-outer / pixel-inner: each prototype plane row is read sequentially.
                    for (int c = 0; c < channels; c++)
                    {
                        float weight = weights[c];
                        float* plane = pProto + (long)c * planeSize;

                        for (int y = 0; y < regionHeight; y++)
                        {
                            float* protoRow = plane + (long)(startY + y) * maskWidth + startX;
                            float* accRow = pAcc + (long)y * regionWidth;

                            int x = 0;

                            if (Avx.IsSupported)
                            {
                                var weightVec = Vector256.Create(weight);
                                int limit = regionWidth - Vector256<float>.Count;

                                for (; x <= limit; x += Vector256<float>.Count)
                                {
                                    var proto = Avx.LoadVector256(protoRow + x);
                                    var acc = Avx.LoadVector256(accRow + x);

                                    acc = Fma.IsSupported
                                        ? Fma.MultiplyAdd(proto, weightVec, acc)
                                        : Avx.Add(acc, Avx.Multiply(proto, weightVec));

                                    Avx.Store(accRow + x, acc);
                                }
                            }

                            for (; x < regionWidth; x++)
                                accRow[x] += protoRow[x] * weight;
                        }
                    }

                    // Activation + write to the Gray8 bitmap.
                    int stride = bitmap.RowBytes;
                    byte* basePtr = (byte*)bitmap.GetPixels().ToPointer();

                    for (int y = 0; y < regionHeight; y++)
                    {
                        byte* row = basePtr + (long)(startY + y) * stride;
                        float* accRow = pAcc + (long)y * regionWidth;

                        for (int x = 0; x < regionWidth; x++)
                        {
                            float value = YoloCore.Sigmoid(accRow[x]);
                            row[startX + x] = (byte)(value * 255); // write directly to Gray8 bitmap
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(accBuffer);
            }
        }
    }
}
