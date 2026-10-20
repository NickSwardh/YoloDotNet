// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Utils
{
    /// <summary>
    /// Maps original image coordinates to segmentation mask (prototype) coordinates:
    /// <c>maskX = imageX * ScaleX + OffsetX</c> and <c>maskY = imageY * ScaleY + OffsetY</c>.
    /// </summary>
    internal readonly record struct MaskTransform(float ScaleX, float OffsetX, float ScaleY, float OffsetY)
    {
        /// <summary>
        /// Creates the transform from the gain and padding returned by <see cref="YoloCore.CalculateGain"/>.
        /// </summary>
        public static MaskTransform Create(
            ImageResize imageResize,
            (float xPad, float yPad, float xGain, float yGain) gain,
            int inputWidth,
            int inputHeight,
            int maskWidth,
            int maskHeight)
        {
            // Image -> model input. Proportional resizing returns gains as original / input,
            // stretched resizing as input / original.
            float imageToInputX = imageResize == ImageResize.Proportional ? 1f / gain.xGain : gain.xGain;
            float imageToInputY = imageResize == ImageResize.Proportional ? 1f / gain.yGain : gain.yGain;

            // Model input -> mask
            float inputToMaskX = (float)maskWidth / inputWidth;
            float inputToMaskY = (float)maskHeight / inputHeight;

            return new MaskTransform(
                imageToInputX * inputToMaskX, gain.xPad * inputToMaskX,
                imageToInputY * inputToMaskY, gain.yPad * inputToMaskY);
        }
    }

    /// <summary>
    /// Builds a detection's segmentation mask, as outlines, by combining the model's mask prototypes with the
    /// detection's mask weights.
    ///
    /// The mask is the bilinearly interpolated mask logit surface, thresholded at the pixel confidence and cropped to
    /// the bounding box; image pixels map to the prototype grid by their centres. Instead of evaluating that surface
    /// for every image pixel, its outline is traced directly on the prototype grid, so the work grows with the mask's
    /// edge rather than its area.
    /// The pixel mask is recovered by filling the outline (see <see cref="MaskRasterizer"/>).
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
        /// Each prototype cell is split into SUBDIVISION × SUBDIVISION bilinearly interpolated sub-cells before tracing,
        /// so the straight outline segments follow the curved mask edge inside a cell closely. Measured on yolo26s-seg:
        /// 2 places the outline 0.044 px (mean) from the exact edge, closer than tracing the full-resolution mask, while
        /// taking about half the time of building a full-resolution bit mask.
        /// </summary>
        private const int SUBDIVISION = 2;

        /// <summary>
        /// Traces the outlines of a detection's mask. The outlines are closed polygons in original image coordinates,
        /// clipped to <paramref name="boundingBox"/>; filling them with the even-odd rule gives the mask, where a pixel
        /// belongs to the mask when its centre lies inside.
        /// </summary>
        /// <param name="prototypes">Mask prototype tensor, laid out as [channels, height, width].</param>
        /// <param name="weights">Per-detection mask weights (one per prototype channel).</param>
        /// <param name="channels">Number of prototype channels to combine.</param>
        /// <param name="maskWidth">Prototype/mask width.</param>
        /// <param name="maskHeight">Prototype/mask height.</param>
        /// <param name="boundingBox">Bounding box in original image coordinates; the mask is cropped to it.</param>
        /// <param name="transform">Maps original image coordinates to mask coordinates.</param>
        /// <param name="pixelConfidence">Minimum pixel confidence for a pixel to be part of the mask.</param>
        public static SKPoint[][] CreateContours(
            ReadOnlySpan<float> prototypes,
            ReadOnlySpan<float> weights,
            int channels,
            int maskWidth,
            int maskHeight,
            SKRectI boundingBox,
            MaskTransform transform,
            double pixelConfidence)
        {
            if (boundingBox.Width <= 0 || boundingBox.Height <= 0 || pixelConfidence >= 1)
                return [];

            var box = (SKRect)boundingBox;

            if (pixelConfidence <= 0)
                return [[new(box.Left, box.Top), new(box.Right, box.Top), new(box.Right, box.Bottom), new(box.Left, box.Bottom)]];

            // sigmoid(logit) > pixelConfidence  <=>  logit > ln(p / (1 - p)), so the outline is traced on the logits
            // without evaluating the sigmoid.
            float threshold = (float)Math.Log(pixelConfidence / (1 - pixelConfidence));

            // Mask cells under the bounding box, plus one cell of margin so the outline reaches the box edges.
            var (startX, _, _) = SamplePosition(boundingBox.Left, transform.ScaleX, transform.OffsetX, maskWidth);
            var (_, endX, _) = SamplePosition(boundingBox.Right - 1, transform.ScaleX, transform.OffsetX, maskWidth);
            var (startY, _, _) = SamplePosition(boundingBox.Top, transform.ScaleY, transform.OffsetY, maskHeight);
            var (_, endY, _) = SamplePosition(boundingBox.Bottom - 1, transform.ScaleY, transform.OffsetY, maskHeight);

            // The margin may reach past the prototype grid (at the image border). Those cells repeat the grid's edge
            // values, as bilinear sampling with clamped edges does, so the outline continues to the box edge and is
            // clipped there.
            startX--;
            startY--;
            endX++;
            endY++;

            int computedStartX = Math.Max(0, startX), computedEndX = Math.Min(maskWidth - 1, endX);
            int computedStartY = Math.Max(0, startY), computedEndY = Math.Min(maskHeight - 1, endY);
            int computedWidth = computedEndX - computedStartX + 1;
            int computedHeight = computedEndY - computedStartY + 1;

            int regionWidth = endX - startX + 1;
            int regionHeight = endY - startY + 1;
            int fieldWidth = (regionWidth - 1) * SUBDIVISION + 1;
            int fieldHeight = (regionHeight - 1) * SUBDIVISION + 1;

            var computed = ArrayPool<float>.Shared.Rent(computedWidth * computedHeight);
            var logits = ArrayPool<float>.Shared.Rent(regionWidth * regionHeight);
            var field = ArrayPool<float>.Shared.Rent(fieldWidth * fieldHeight);

            try
            {
                AccumulateLogits(computed, prototypes, weights, channels, maskWidth, maskHeight, computedStartX, computedStartY, computedWidth, computedHeight);

                for (int y = 0; y < regionHeight; y++)
                {
                    int sourceRow = (Math.Clamp(startY + y, computedStartY, computedEndY) - computedStartY) * computedWidth;

                    for (int x = 0; x < regionWidth; x++)
                        logits[y * regionWidth + x] = computed[sourceRow + Math.Clamp(startX + x, computedStartX, computedEndX) - computedStartX];
                }

                Subdivide(logits, regionWidth, regionHeight, SUBDIVISION, field);

                // Field sample (i, j) lies at mask position (startX + i / s, startY + j / s), and mask position m at
                // image position (m + 0.5 - offset) / scale (the inverse of the pixel-centre mapping).
                var contours = MaskContourTracer.TraceField(
                    field.AsSpan(0, fieldWidth * fieldHeight), fieldWidth, fieldHeight, threshold,
                    scaleX: 1f / (SUBDIVISION * transform.ScaleX), offsetX: (startX + 0.5f - transform.OffsetX) / transform.ScaleX,
                    scaleY: 1f / (SUBDIVISION * transform.ScaleY), offsetY: (startY + 0.5f - transform.OffsetY) / transform.ScaleY);

                // The mask is cropped to the bounding box.
                var clipped = new List<SKPoint[]>(contours.Count);

                foreach (var contour in contours)
                {
                    var polygon = MaskContourTracer.ClipToRectangle(contour, box);

                    if (polygon.Length >= 3)
                        clipped.Add(polygon);
                }

                return [.. clipped];
            }
            finally
            {
                ArrayPool<float>.Shared.Return(computed);
                ArrayPool<float>.Shared.Return(logits);
                ArrayPool<float>.Shared.Return(field);
            }
        }

        /// <summary>
        /// Bilinearly interpolates a field onto a grid <paramref name="subdivision"/> times finer (sharing its corner samples).
        /// </summary>
        private static void Subdivide(float[] source, int width, int height, int subdivision, float[] destination)
        {
            int fineWidth = (width - 1) * subdivision + 1;
            int fineHeight = (height - 1) * subdivision + 1;
            float inverse = 1f / subdivision;
            int lastCellX = Math.Max(0, width - 2);
            int lastCellY = Math.Max(0, height - 2);

            for (int fy = 0; fy < fineHeight; fy++)
            {
                int y0 = Math.Min(fy / subdivision, lastCellY);
                int y1 = Math.Min(y0 + 1, height - 1);
                float wy = fy * inverse - y0;
                int row = fy * fineWidth;

                for (int fx = 0; fx < fineWidth; fx++)
                {
                    int x0 = Math.Min(fx / subdivision, lastCellX);
                    int x1 = Math.Min(x0 + 1, width - 1);
                    float wx = fx * inverse - x0;

                    float top = source[y0 * width + x0] + (source[y0 * width + x1] - source[y0 * width + x0]) * wx;
                    float bottom = source[y1 * width + x0] + (source[y1 * width + x1] - source[y1 * width + x0]) * wx;
                    destination[row + fx] = top + (bottom - top) * wy;
                }
            }
        }

        /// <summary>
        /// Maps the centre of image pixel <paramref name="pixel"/> to the mask grid and returns the two
        /// neighbouring mask cells and the interpolation weight of the second one (edges are clamped).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (int index0, int index1, float weight) SamplePosition(int pixel, float scale, float offset, int size)
        {
            float position = Math.Clamp((pixel + 0.5f) * scale + offset - 0.5f, 0, size - 1);
            int index0 = (int)position;

            return (index0, Math.Min(index0 + 1, size - 1), position - index0);
        }

        /// <summary>
        /// Accumulates the weighted prototype sum (mask logits) for the given mask region.
        /// </summary>
        private static void AccumulateLogits(
            float[] accBuffer,
            ReadOnlySpan<float> prototypes,
            ReadOnlySpan<float> weights,
            int channels,
            int maskWidth,
            int maskHeight,
            int startX,
            int startY,
            int regionWidth,
            int regionHeight)
        {
            int planeSize = maskWidth * maskHeight;

            Array.Clear(accBuffer, 0, regionWidth * regionHeight);

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
            }
        }
    }
}
