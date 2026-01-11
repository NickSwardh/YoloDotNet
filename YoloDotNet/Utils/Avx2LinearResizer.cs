// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Utils
{
    /// <summary>
    /// Resize a Gray8 image using bilinear interpolation with AVX2 acceleration.
    /// Source and destination must be Gray8 format.
    /// 
    /// The method relies heavily on SIMD (Single Instruction, Multiple Data)
    /// for processing multiple pixels at once, using a single instruction.
    /// 
    /// Uses AVX2 (Advanced Vector Extensions 2) that gives the CPU the ability
    /// to do 256-bit wide SIMD math — super fast math on multiple values at once.
    /// </summary>
    public static unsafe class Avx2LinearResizer
    {
        public static void ScalePixels(SKBitmap src, SKBitmap dst)
        {
            if (src.ColorType != SKColorType.Gray8 || dst.ColorType != SKColorType.Gray8)
                throw new YoloDotNetException("Both source and destination bitmaps must be Gray8 format.");

            if (!Avx2.IsSupported)
                throw new PlatformNotSupportedException("AVX2 is required.");

            int srcW = src.Width;
            int srcH = src.Height;
            int dstW = dst.Width;
            int dstH = dst.Height;

            Span<byte> srcSpan = src.GetPixelSpan();
            Span<byte> dstSpan = dst.GetPixelSpan();

            int srcStride = src.RowBytes;
            int dstStride = dst.RowBytes;

            float scaleX = (float)srcW / dstW;
            float scaleY = (float)srcH / dstH;

            // Precompute horizontal mapping
            int[] x0s = new int[dstW];
            int[] x1s = new int[dstW];
            float[] wxs = new float[dstW];

            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = dx * scaleX;
                int x0 = (int)sx;
                int x1 = Math.Min(x0 + 1, srcW - 1);
                x0s[dx] = x0;
                x1s[dx] = x1;
                wxs[dx] = sx - x0;
            }

            fixed (byte* pSrc = srcSpan)
            fixed (byte* pDst = dstSpan)
            fixed (int* px0s = x0s)
            fixed (int* px1s = x1s)
            fixed (float* pwxs = wxs)
            {
                for (int dy = 0; dy < dstH; dy++)
                {
                    float sy = dy * scaleY;
                    int y0 = (int)sy;
                    int y1 = Math.Min(y0 + 1, srcH - 1);
                    float wy = sy - y0;

                    int srcRow0 = y0 * srcStride;
                    int srcRow1 = y1 * srcStride;
                    int dstRow = dy * dstStride;

                    var wyVec = Vector256.Create(wy);

                    int dx = 0;
                    while (dx + 32 <= dstW)
                    {
                        // Check if x0 and x1 are contiguous for this 32-pixel block
                        if (IsContiguous(px0s + dx, 32) && IsContiguous(px1s + dx, 32))
                        {
                            // SIMD path
                            byte* topLeft = pSrc + srcRow0 + px0s[dx];
                            byte* topRight = pSrc + srcRow0 + px1s[dx];
                            byte* bottomLeft = pSrc + srcRow1 + px0s[dx];
                            byte* bottomRight = pSrc + srcRow1 + px1s[dx];

                            Vector256<byte> p00 = Avx.LoadVector256(topLeft);
                            Vector256<byte> p10 = Avx.LoadVector256(topRight);
                            Vector256<byte> p01 = Avx.LoadVector256(bottomLeft);
                            Vector256<byte> p11 = Avx.LoadVector256(bottomRight);

                            var wx0 = Avx.LoadVector256(pwxs + dx);
                            var wx1 = Avx.LoadVector256(pwxs + dx + 8);
                            var wx2 = Avx.LoadVector256(pwxs + dx + 16);
                            var wx3 = Avx.LoadVector256(pwxs + dx + 24);

                            var p00f = BytesToFloats(p00);
                            var p10f = BytesToFloats(p10);
                            var p01f = BytesToFloats(p01);
                            var p11f = BytesToFloats(p11);

                            var iTop = new Vector256<float>[4];
                            var iBottom = new Vector256<float>[4];

                            for (int i = 0; i < 4; i++)
                            {
                                var deltaTop = Avx.Subtract(p10f[i], p00f[i]);
                                var deltaBottom = Avx.Subtract(p11f[i], p01f[i]);
                                var wxVec = i switch
                                {
                                    0 => wx0,
                                    1 => wx1,
                                    2 => wx2,
                                    3 => wx3,
                                    _ => wx0
                                };

                                iTop[i] = Fma.IsSupported
                                    ? Fma.MultiplyAdd(deltaTop, wxVec, p00f[i])
                                    : Avx.Add(p00f[i], Avx.Multiply(deltaTop, wxVec));

                                iBottom[i] = Fma.IsSupported
                                    ? Fma.MultiplyAdd(deltaBottom, wxVec, p01f[i])
                                    : Avx.Add(p01f[i], Avx.Multiply(deltaBottom, wxVec));
                            }

                            for (int i = 0; i < 4; i++)
                            {
                                var delta = Avx.Subtract(iBottom[i], iTop[i]);
                                var interp = Fma.IsSupported
                                    ? Fma.MultiplyAdd(delta, wyVec, iTop[i])
                                    : Avx.Add(iTop[i], Avx.Multiply(delta, wyVec));

                                Store8FloatsToBytes(interp, pDst + dstRow + dx + i * 8);
                            }

                            dx += 32;
                        }
                        else
                        {
                            // Non-contiguous: fallback to scalar
                            for (int i = 0; i < 32; i++)
                            {
                                int px = dx + i;
                                int x0 = px0s[px];
                                int x1 = px1s[px];
                                float wx = pwxs[px];

                                byte p00 = pSrc[srcRow0 + x0];
                                byte p10 = pSrc[srcRow0 + x1];
                                byte p01 = pSrc[srcRow1 + x0];
                                byte p11 = pSrc[srcRow1 + x1];

                                float iTop = p00 + (p10 - p00) * wx;
                                float iBottom = p01 + (p11 - p01) * wx;
                                float iValue = iTop + (iBottom - iTop) * wy;

                                pDst[dstRow + px] = (byte)(iValue + 0.5f);
                            }
                            dx += 32;
                        }
                    }

                    // Remaining pixels
                    for (; dx < dstW; dx++)
                    {
                        int x0 = px0s[dx];
                        int x1 = px1s[dx];
                        float wx = pwxs[dx];

                        byte p00 = pSrc[srcRow0 + x0];
                        byte p10 = pSrc[srcRow0 + x1];
                        byte p01 = pSrc[srcRow1 + x0];
                        byte p11 = pSrc[srcRow1 + x1];

                        float iTop = p00 + (p10 - p00) * wx;
                        float iBottom = p01 + (p11 - p01) * wx;
                        float iValue = iTop + (iBottom - iTop) * wy;

                        pDst[dstRow + dx] = (byte)(iValue + 0.5f);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsContiguous(int* ptr, int length)
        {
            for (int i = 1; i < length; i++)
                if (ptr[i] != ptr[i - 1] + 1)
                    return false;
            return true;
        }

        private static Vector256<float>[] BytesToFloats(Vector256<byte> vec)
        {
            // Split 256-bit vector into 2 x 128-bit lanes
            Vector128<byte> low = vec.GetLower();
            Vector128<byte> high = vec.GetUpper();

            // Unpack bytes to ushort (zero-extend)
            Vector128<ushort> lowLo = Sse2.UnpackLow(low, Vector128<byte>.Zero).AsUInt16();
            Vector128<ushort> lowHi = Sse2.UnpackHigh(low, Vector128<byte>.Zero).AsUInt16();
            Vector128<ushort> highLo = Sse2.UnpackLow(high, Vector128<byte>.Zero).AsUInt16();
            Vector128<ushort> highHi = Sse2.UnpackHigh(high, Vector128<byte>.Zero).AsUInt16();

            // Unpack ushort to uint (zero-extend)
            Vector128<uint> lowLoLo = Sse2.UnpackLow(lowLo, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> lowLoHi = Sse2.UnpackHigh(lowLo, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> lowHiLo = Sse2.UnpackLow(lowHi, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> lowHiHi = Sse2.UnpackHigh(lowHi, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> highLoLo = Sse2.UnpackLow(highLo, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> highLoHi = Sse2.UnpackHigh(highLo, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> highHiLo = Sse2.UnpackLow(highHi, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> highHiHi = Sse2.UnpackHigh(highHi, Vector128<ushort>.Zero).AsUInt32();

            // Combine pairs into 256-bit vectors
            Vector256<uint> vec0 = Vector256.Create(lowLoLo, lowLoHi);
            Vector256<uint> vec1 = Vector256.Create(lowHiLo, lowHiHi);
            Vector256<uint> vec2 = Vector256.Create(highLoLo, highLoHi);
            Vector256<uint> vec3 = Vector256.Create(highHiLo, highHiHi);

            // Convert uint to float vectors
            Vector256<float> f0 = Avx.ConvertToVector256Single(vec0.AsInt32());
            Vector256<float> f1 = Avx.ConvertToVector256Single(vec1.AsInt32());
            Vector256<float> f2 = Avx.ConvertToVector256Single(vec2.AsInt32());
            Vector256<float> f3 = Avx.ConvertToVector256Single(vec3.AsInt32());

            return [f0, f1, f2, f3];
        }

        private static void Store8FloatsToBytes(Vector256<float> vec, byte* dst)
        {
            var rounded = Avx.Add(vec, Vector256.Create(0.5f));
            var intVec = Avx.ConvertToVector256Int32WithTruncation(rounded);

            Vector128<int> lower = intVec.GetLower();
            Vector128<int> upper = intVec.GetUpper();

            Vector128<short> packed16 = Sse2.PackSignedSaturate(lower, upper);
            Vector128<byte> packed8 = Sse2.PackUnsignedSaturate(packed16, packed16);

            for (int i = 0; i < 8; i++)
                dst[i] = packed8.GetElement(i);
        }
    }
}
