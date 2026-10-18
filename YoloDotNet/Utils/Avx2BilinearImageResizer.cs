// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Utils
{
    /// <summary>
    /// Bilinear resize of opaque 32-bit BGRA/RGBA images into the model's RGBX input buffer, using AVX2.
    /// </summary>
    /// <remarks>
    /// A separable bilinear filter with pixel-centre sampling (<c>src = (dst + 0.5) * scale - 0.5</c>), clamped edges and
    /// 11-bit fixed-point weights that are rounded once, at the end. This is the same arithmetic as the bilinear resize
    /// YOLO models are trained with, and it matches SkiaSharp's linear filter to within 1 in rare pixels.
    /// <para>
    /// Each source row is filtered horizontally once and cached, then pairs of filtered rows are blended vertically and
    /// packed straight into the destination. About 3x faster than drawing the image through an <see cref="SKCanvas"/>.
    /// </para>
    /// </remarks>
    internal static unsafe class Avx2BilinearImageResizer
    {
        private const int COEFFICIENT_BITS = 11;
        private const int COEFFICIENT_SCALE = 1 << COEFFICIENT_BITS;   // the two weights of one axis sum to 2048
        private const int SHIFT = COEFFICIENT_BITS * 2;                // both axes together: 2048 * 2048
        private const int ROUNDING = 1 << (SHIFT - 1);

        public static bool IsSupported => Avx2.IsSupported;

        /// <summary>
        /// Resizes <paramref name="source"/> into <paramref name="destination"/> of the RGBX buffer at <paramref name="buffer"/>.
        /// </summary>
        /// <returns>False, without writing anything, for inputs this resizer does not handle. The caller then uses SkiaSharp.</returns>
        public static bool TryResize(SKPixmap source, IntPtr buffer, int bufferRowBytes, SKRectI destination)
        {
            if (!IsSupported)
                return false;

            var colorType = source.ColorType;
            bool isBgra = colorType == SKColorType.Bgra8888;

            if (!isBgra && colorType != SKColorType.Rgba8888)
                return false;

            // The alpha channel is not used. Callers pass 32-bit images labelled opaque, so their stored colours are read
            // as they are (see ImageResizeExtension.IgnoreAlpha).
            if (source.AlphaType != SKAlphaType.Opaque)
                return false;

            int srcWidth = source.Width;
            int srcHeight = source.Height;
            int dstWidth = destination.Width;
            int dstHeight = destination.Height;

            if (srcWidth < 2 || srcHeight < 2 || dstWidth < 1 || dstHeight < 1)
                return false;

            var pool = ArrayPool<int>.Shared;
            var xOffsets = pool.Rent(dstWidth);
            var xWeights = pool.Rent(dstWidth * 8);
            var yRows = pool.Rent(dstHeight);
            var yWeights = pool.Rent(dstHeight * 2);
            var rowA = pool.Rent(dstWidth * 4);
            var rowB = pool.Rent(dstWidth * 4);

            try
            {
                // Per destination column: byte offset of the left source pixel, and the weights of the pixel pair,
                // repeated for the 4 channels of each pixel.
                for (int x = 0; x < dstWidth; x++)
                {
                    var (first, weight) = SamplePosition(x, srcWidth, dstWidth);
                    xOffsets[x] = first * 4;

                    for (int c = 0; c < 4; c++)
                    {
                        xWeights[x * 8 + c] = COEFFICIENT_SCALE - weight;
                        xWeights[x * 8 + 4 + c] = weight;
                    }
                }

                // Per destination row: the upper source row, and the weights of the row pair.
                for (int y = 0; y < dstHeight; y++)
                {
                    var (first, weight) = SamplePosition(y, srcHeight, dstHeight);
                    yRows[y] = first;
                    yWeights[y * 2] = COEFFICIENT_SCALE - weight;
                    yWeights[y * 2 + 1] = weight;
                }

                byte* src = (byte*)source.GetPixels();
                int srcRowBytes = source.RowBytes;
                byte* dst = (byte*)buffer + destination.Top * bufferRowBytes + destination.Left * 4;

                fixed (int* pXOffsets = xOffsets, pXWeights = xWeights, pRowA = rowA, pRowB = rowB)
                {
                    // Two cached, horizontally filtered source rows. When upscaling, neighbouring destination rows share
                    // source rows, so most rows are filtered once.
                    int* cacheA = pRowA, cacheB = pRowB;
                    int rowInA = -1, rowInB = -1;

                    for (int y = 0; y < dstHeight; y++)
                    {
                        int upper = yRows[y];
                        int lower = upper + 1;

                        // Keep the upper row in A and the lower row in B, filtering only rows not already cached.
                        if (rowInB == upper)
                        {
                            int* cache = cacheA;
                            cacheA = cacheB;
                            cacheB = cache;

                            int row = rowInA;
                            rowInA = rowInB;
                            rowInB = row;
                        }

                        if (rowInA != upper)
                        {
                            FilterRow(src + upper * srcRowBytes, pXOffsets, pXWeights, cacheA, dstWidth);
                            rowInA = upper;
                        }

                        if (rowInB != lower)
                        {
                            FilterRow(src + lower * srcRowBytes, pXOffsets, pXWeights, cacheB, dstWidth);
                            rowInB = lower;
                        }

                        BlendRows(cacheA, cacheB, yWeights[y * 2], yWeights[y * 2 + 1], dst + y * bufferRowBytes, dstWidth, isBgra);
                    }
                }

                return true;
            }
            finally
            {
                pool.Return(xOffsets);
                pool.Return(xWeights);
                pool.Return(yRows);
                pool.Return(yWeights);
                pool.Return(rowA);
                pool.Return(rowB);
            }
        }

        /// <summary>
        /// The pair of source pixels a destination pixel is sampled between: the first source index, and the weight of
        /// the second one in 1/2048 units. The pair always lies inside the source, so the edges are clamped.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (int First, int Weight) SamplePosition(int index, int srcSize, int dstSize)
        {
            float position = (float)((index + 0.5) * ((double)srcSize / dstSize) - 0.5);
            int first = (int)MathF.Floor(position);
            float fraction = position - first;

            if (first < 0)
            {
                first = 0;
                fraction = 0;
            }

            // The last source pixel, expressed as the pair (size - 2, size - 1) with all weight on the second.
            if (first >= srcSize - 1)
            {
                first = srcSize - 2;
                fraction = 1;
            }

            return (first, (int)MathF.Round(fraction * COEFFICIENT_SCALE, MidpointRounding.ToEven));
        }

        /// <summary>
        /// Filters one source row horizontally: per output pixel and channel, <c>p0 * (2048 - w) + p1 * w</c> as 32-bit integers.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void FilterRow(byte* row, int* xOffsets, int* xWeights, int* output, int count)
        {
            for (int x = 0; x < count; x++)
            {
                // The two source pixels are adjacent: one 8-byte load, widened to 8 ints (2 pixels x 4 channels).
                var pair = Vector128.CreateScalarUnsafe(*(long*)(row + xOffsets[x])).AsByte();
                var weighted = Avx2.MultiplyLow(Avx2.ConvertToVector256Int32(pair), Avx.LoadVector256(xWeights + x * 8));

                Sse2.Store(output + x * 4, Sse2.Add(weighted.GetLower(), weighted.GetUpper()));
            }
        }

        /// <summary>
        /// Blends two filtered rows into RGBX bytes: <c>(h0 * w0 + h1 * w1 + 2^21) &gt;&gt; 22</c>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void BlendRows(int* upper, int* lower, int upperWeight, int lowerWeight, byte* output, int count, bool isBgra)
        {
            var w0 = Vector128.Create(upperWeight);
            var w1 = Vector128.Create(lowerWeight);
            var rounding = Vector128.Create(ROUNDING);

            // The destination stores R, G, B, X. BGRA sources swap their first and third byte; RGBA is already in order.
            var channelOrder = isBgra
                ? Vector128.Create((byte)2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15)
                : Vector128.Create((byte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

            int x = 0;

            // 4 pixels per iteration: 4 x 4 channels, packed to 16 bytes.
            for (; x <= count - 4; x += 4)
            {
                int i = x * 4;
                var p0 = Blend(upper + i, lower + i, w0, w1, rounding);
                var p1 = Blend(upper + i + 4, lower + i + 4, w0, w1, rounding);
                var p2 = Blend(upper + i + 8, lower + i + 8, w0, w1, rounding);
                var p3 = Blend(upper + i + 12, lower + i + 12, w0, w1, rounding);

                var bytes = Sse2.PackUnsignedSaturate(Sse2.PackSignedSaturate(p0, p1), Sse2.PackSignedSaturate(p2, p3));
                Sse2.Store(output + i, Ssse3.Shuffle(bytes, channelOrder));
            }

            for (; x < count; x++)
            {
                var pixel = Blend(upper + x * 4, lower + x * 4, w0, w1, rounding);
                var packed = Sse2.PackSignedSaturate(pixel, pixel);
                var bytes = Sse2.PackUnsignedSaturate(packed, packed);

                *(uint*)(output + x * 4) = Ssse3.Shuffle(bytes, channelOrder).AsUInt32().ToScalar();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<int> Blend(int* upper, int* lower, Vector128<int> upperWeight, Vector128<int> lowerWeight, Vector128<int> rounding)
        {
            // At most 255 * 2048 * 2048 + 2^21, which fits in a 32-bit integer.
            var sum = Sse2.Add(
                Sse41.MultiplyLow(Sse2.LoadVector128(upper), upperWeight),
                Sse41.MultiplyLow(Sse2.LoadVector128(lower), lowerWeight));

            return Sse2.ShiftRightArithmetic(Sse2.Add(sum, rounding), SHIFT);
        }
    }
}
