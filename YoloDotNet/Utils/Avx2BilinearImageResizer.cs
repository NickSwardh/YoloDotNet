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
    /// YOLO models are trained with.
    /// <para>
    /// Each source row is filtered horizontally once and cached, then pairs of filtered rows are blended vertically and
    /// packed straight into the destination.
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
            => TryResizeCore(source, buffer, bufferRowBytes, destination, null, 0);

        /// <summary>
        /// Writes the same rounded pixels directly into normalized planar RGB, without an intermediate RGBX image.
        /// </summary>
        internal static bool TryResizeNormalized(SKPixmap source, float[] tensor, int width, int height, SKRectI destination)
        {
            fixed (float* output = tensor)
                return TryResizeCore(source, IntPtr.Zero, width * sizeof(float), destination, output, height);
        }

        /// <summary>
        /// Writes the same rounded pixels directly into normalized planar RGB half-precision floats, converting each
        /// channel value with <paramref name="byteToHalf"/> (the normalized half for every byte value).
        /// </summary>
        internal static bool TryResizeNormalizedHalf(SKPixmap source, ushort[] tensor, int width, int height, SKRectI destination, ushort[] byteToHalf)
        {
            fixed (ushort* output = tensor)
            fixed (ushort* table = byteToHalf)
                return TryResizeCore(source, IntPtr.Zero, width * sizeof(ushort), destination, null, height, output, table);
        }

        private static bool TryResizeCore(SKPixmap source, IntPtr buffer, int bufferRowBytes, SKRectI destination,
            float* tensor, int tensorHeight, ushort* halfTensor = null, ushort* byteToHalf = null)
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

            if (tensor != null)
                FillPadding(tensor, bufferRowBytes / sizeof(float), tensorHeight, destination);
            else if (halfTensor != null)
                FillPaddingHalf(halfTensor, bufferRowBytes / sizeof(ushort), tensorHeight, destination, byteToHalf);

            var pool = ArrayPool<int>.Shared;
            var xOffsets = pool.Rent(dstWidth);
            var xWeights = pool.Rent(dstWidth * 8);
            var yRows = pool.Rent(dstHeight);
            var yWeights = pool.Rent(dstHeight * 2);

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

                fixed (int* pXOffsets = xOffsets, pXWeights = xWeights, pYRows = yRows, pYWeights = yWeights)
                {
                    var rows = new RowJob(src, srcRowBytes, pXOffsets, pXWeights, pYRows, pYWeights, dstWidth, isBgra,
                        dst, bufferRowBytes, tensor, halfTensor, byteToHalf, tensorHeight, destination);

                    // Large images are split into horizontal bands resized in parallel. Every row is computed exactly as
                    // in one pass; a band only re-filters the source rows at its top edge. Parallel.For returns when all
                    // bands are done, so the pinned arrays above stay valid for them.
                    int bands = RowBands(dstWidth, dstHeight);

                    if (bands == 1)
                        ResizeRows(rows, 0, dstHeight);
                    else
                        Parallel.For(0, bands, band => ResizeRows(rows, dstHeight * band / bands, dstHeight * (band + 1) / bands));
                }

                return true;
            }
            finally
            {
                pool.Return(xOffsets);
                pool.Return(xWeights);
                pool.Return(yRows);
                pool.Return(yWeights);
            }
        }

        /// <summary>
        /// The number of bands to resize in parallel: 1 for small images, where the threading overhead outweighs the work.
        /// </summary>
        private static int RowBands(int dstWidth, int dstHeight)
        {
            if ((long)dstWidth * dstHeight < MIN_PIXELS_PER_BAND * 2)
                return 1;

            int bands = Math.Min(MAX_BANDS, Environment.ProcessorCount / 2);
            return (int)Math.Clamp(Math.Min(bands, (long)dstWidth * dstHeight / MIN_PIXELS_PER_BAND), 1, MAX_BANDS);
        }

        private const int MAX_BANDS = 4;
        private const int MIN_PIXELS_PER_BAND = 64 * 1024;

        /// <summary>Everything one band needs; pointers stay valid while <see cref="TryResizeCore"/> runs.</summary>
        private readonly struct RowJob(byte* src, int srcRowBytes, int* xOffsets, int* xWeights, int* yRows, int* yWeights,
            int dstWidth, bool isBgra, byte* dst, int bufferRowBytes, float* tensor, ushort* halfTensor, ushort* byteToHalf,
            int tensorHeight, SKRectI destination)
        {
            public readonly byte* Src = src;
            public readonly int SrcRowBytes = srcRowBytes;
            public readonly int* XOffsets = xOffsets;
            public readonly int* XWeights = xWeights;
            public readonly int* YRows = yRows;
            public readonly int* YWeights = yWeights;
            public readonly int DstWidth = dstWidth;
            public readonly bool IsBgra = isBgra;
            public readonly byte* Dst = dst;
            public readonly int BufferRowBytes = bufferRowBytes;
            public readonly float* Tensor = tensor;
            public readonly ushort* HalfTensor = halfTensor;
            public readonly ushort* ByteToHalf = byteToHalf;
            public readonly int TensorHeight = tensorHeight;
            public readonly SKRectI Destination = destination;
        }

        /// <summary>
        /// Resizes destination rows <paramref name="yStart"/> to <paramref name="yEnd"/> (exclusive) into the RGBX buffer or tensor.
        /// </summary>
        private static void ResizeRows(in RowJob job, int yStart, int yEnd)
        {
            var pool = ArrayPool<int>.Shared;
            var rowA = pool.Rent(job.DstWidth * 4);
            var rowB = pool.Rent(job.DstWidth * 4);

            try
            {
                byte* src = job.Src;
                int srcRowBytes = job.SrcRowBytes;
                int* pXOffsets = job.XOffsets, pXWeights = job.XWeights, yRows = job.YRows, yWeights = job.YWeights;
                int dstWidth = job.DstWidth, bufferRowBytes = job.BufferRowBytes, tensorHeight = job.TensorHeight;
                bool isBgra = job.IsBgra;
                byte* dst = job.Dst;
                float* tensor = job.Tensor;
                ushort* halfTensor = job.HalfTensor, byteToHalf = job.ByteToHalf;
                var destination = job.Destination;

                fixed (int* pRowA = rowA, pRowB = rowB)
                {
                    // Two cached, horizontally filtered source rows. When upscaling, neighbouring destination rows share
                    // source rows, so most rows are filtered once.
                    int* cacheA = pRowA, cacheB = pRowB;
                    int rowInA = -1, rowInB = -1;

                    for (int y = yStart; y < yEnd; y++)
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

                        if (tensor != null)
                        {
                            int width = bufferRowBytes / sizeof(float);
                            BlendNormalizedRows(cacheA, cacheB, yWeights[y * 2], yWeights[y * 2 + 1],
                                tensor + (destination.Top + y) * width + destination.Left, width * tensorHeight, dstWidth, isBgra);
                        }
                        else if (halfTensor != null)
                        {
                            int width = bufferRowBytes / sizeof(ushort);
                            BlendNormalizedHalfRows(cacheA, cacheB, yWeights[y * 2], yWeights[y * 2 + 1],
                                halfTensor + (destination.Top + y) * width + destination.Left, width * tensorHeight, dstWidth, isBgra, byteToHalf);
                        }
                        else
                            BlendRows(cacheA, cacheB, yWeights[y * 2], yWeights[y * 2 + 1], dst + y * bufferRowBytes, dstWidth, isBgra);
                    }
                }
            }
            finally
            {
                pool.Return(rowA);
                pool.Return(rowB);
            }
        }

        private static void FillPaddingHalf(ushort* tensor, int width, int height, SKRectI destination, ushort* byteToHalf)
        {
            var color = ImageConfig.LetterboxPaddingColor;
            for (int channel = 0; channel < 3; channel++)
            {
                ushort value = byteToHalf[channel == 0 ? color.Red : channel == 1 ? color.Green : color.Blue];
                ushort* plane = tensor + channel * width * height;
                new Span<ushort>(plane, destination.Top * width).Fill(value);
                new Span<ushort>(plane + destination.Bottom * width, (height - destination.Bottom) * width).Fill(value);
                if (destination.Left == 0 && destination.Right == width)
                    continue;

                for (int y = destination.Top; y < destination.Bottom; y++)
                {
                    new Span<ushort>(plane + y * width, destination.Left).Fill(value);
                    new Span<ushort>(plane + y * width + destination.Right, width - destination.Right).Fill(value);
                }
            }
        }

        private static void BlendNormalizedHalfRows(int* upper, int* lower, int upperWeight, int lowerWeight,
            ushort* output, int planeSize, int count, bool isBgra, ushort* byteToHalf)
        {
            var w0 = Vector128.Create(upperWeight);
            var w1 = Vector128.Create(lowerWeight);
            var rounding = Vector128.Create(ROUNDING);
            ushort* first = output + (isBgra ? planeSize * 2 : 0);
            ushort* green = output + planeSize;
            ushort* third = output + (isBgra ? 0 : planeSize * 2);

            // Blend yields 0-255 (non-negative weights that sum to one), the same values BlendRows packs into bytes,
            // so the table gives exactly what normalizing the RGBX image would. The byte cast only bounds the index.
            for (int x = 0; x < count; x++)
            {
                var pixel = Blend(upper + x * 4, lower + x * 4, w0, w1, rounding);
                first[x] = byteToHalf[(byte)pixel.GetElement(0)];
                green[x] = byteToHalf[(byte)pixel.GetElement(1)];
                third[x] = byteToHalf[(byte)pixel.GetElement(2)];
            }
        }

        private static void FillPadding(float* tensor, int width, int height, SKRectI destination)
        {
            var color = ImageConfig.LetterboxPaddingColor;
            for (int channel = 0; channel < 3; channel++)
            {
                float value = (channel == 0 ? color.Red : channel == 1 ? color.Green : color.Blue) * (1f / 255f);
                float* plane = tensor + channel * width * height;
                new Span<float>(plane, destination.Top * width).Fill(value);
                new Span<float>(plane + destination.Bottom * width, (height - destination.Bottom) * width).Fill(value);
                if (destination.Left == 0 && destination.Right == width)
                    continue;

                for (int y = destination.Top; y < destination.Bottom; y++)
                {
                    new Span<float>(plane + y * width, destination.Left).Fill(value);
                    new Span<float>(plane + y * width + destination.Right, width - destination.Right).Fill(value);
                }
            }
        }

        private static void BlendNormalizedRows(int* upper, int* lower, int upperWeight, int lowerWeight,
            float* output, int planeSize, int count, bool isBgra)
        {
            var w0 = Vector128.Create(upperWeight);
            var w1 = Vector128.Create(lowerWeight);
            var rounding = Vector128.Create(ROUNDING);
            var scale = Vector128.Create(1f / 255f);
            float* first = output + (isBgra ? planeSize * 2 : 0);
            float* green = output + planeSize;
            float* third = output + (isBgra ? 0 : planeSize * 2);

            int x = 0;
            for (; x <= count - 4; x += 4)
            {
                var p0 = Blend(upper + x * 4, lower + x * 4, w0, w1, rounding);
                var p1 = Blend(upper + x * 4 + 4, lower + x * 4 + 4, w0, w1, rounding);
                var p2 = Blend(upper + x * 4 + 8, lower + x * 4 + 8, w0, w1, rounding);
                var p3 = Blend(upper + x * 4 + 12, lower + x * 4 + 12, w0, w1, rounding);
                var low01 = Sse2.UnpackLow(p0, p1).AsInt64();
                var low23 = Sse2.UnpackLow(p2, p3).AsInt64();
                var c0 = Sse2.UnpackLow(low01, low23).AsInt32();
                var c1 = Sse2.UnpackHigh(low01, low23).AsInt32();
                var c2 = Sse2.UnpackLow(Sse2.UnpackHigh(p0, p1).AsInt64(), Sse2.UnpackHigh(p2, p3).AsInt64()).AsInt32();
                Sse.Store(first + x, Sse.Multiply(Sse2.ConvertToVector128Single(c0), scale));
                Sse.Store(green + x, Sse.Multiply(Sse2.ConvertToVector128Single(c1), scale));
                Sse.Store(third + x, Sse.Multiply(Sse2.ConvertToVector128Single(c2), scale));
            }

            for (; x < count; x++)
            {
                var pixel = Blend(upper + x * 4, lower + x * 4, w0, w1, rounding);
                first[x] = pixel.GetElement(0) * (1f / 255f);
                green[x] = pixel.GetElement(1) * (1f / 255f);
                third[x] = pixel.GetElement(2) * (1f / 255f);
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
