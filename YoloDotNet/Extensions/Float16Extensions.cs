// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    /// <summary>
    /// Fast conversion of half-precision (FP16) tensor data to single precision (FP32).
    /// </summary>
    public static class Float16Extensions
    {
        // Bit-level constants for widening an IEEE 754 binary16 value into binary32.
        private const int EXPONENT_AND_MANTISSA_MASK = 0x7FFF;
        private const int SIGN_MASK = 0x8000;
        private const int MANTISSA_MASK = 0x03FF;
        private const int SHIFTED_EXPONENT_MASK = 0x7C00 << 13;   // half exponent bits, moved into float position
        private const int EXPONENT_REBIAS = (127 - 15) << 23;     // float exponent bias minus half exponent bias
        private const int INF_OR_NAN_REBIAS = (128 - 16) << 23;   // extra offset that turns exponent 31 into 255
        private const int SUBNORMAL_MAGIC = 113 << 23;            // 2^-14 as float bits
        private const int QUIET_NAN_BIT = 0x00400000;

        /// <summary>
        /// Converts half-precision values to single precision.
        /// </summary>
        /// <remarks>
        /// The result is bit-identical to casting every <see cref="Half"/> to <see cref="float"/>, for all 65,536 possible
        /// values, but without a slow path for subnormal values. Subnormals make up most of a model's output (class scores
        /// close to zero), so this matters for FP16 models: converting a 640x640 detection output takes about 0.14 ms here,
        /// against about 0.85 ms with a per-value cast.
        /// <para>
        /// Based on the public-domain <c>half_to_float_fast4</c> by Fabian "ryg" Giesen
        /// (https://gist.github.com/rygorous/2144712), with the branches replaced by vector masks so that several values
        /// are converted per instruction.
        /// </para>
        /// </remarks>
        /// <param name="source">The half-precision values.</param>
        /// <param name="destination">Receives the single-precision values. Must be at least as long as <paramref name="source"/>.</param>
        public static unsafe void ConvertToSingle(this ReadOnlySpan<Half> source, Span<float> destination)
        {
            if (destination.Length < source.Length)
                throw new ArgumentException("The destination is shorter than the source.", nameof(destination));

            int length = source.Length;
            int i = 0;

            fixed (Half* halfSource = source)
            fixed (float* dst = destination)
            {
                ushort* src = (ushort*)halfSource;

                if (Vector256.IsHardwareAccelerated)
                {
                    // 16 halves per iteration: one load, widened into two vectors of 8.
                    for (; i <= length - Vector256<ushort>.Count; i += Vector256<ushort>.Count)
                    {
                        var halves = Vector256.Load(src + i);
                        Widen(Vector256.WidenLower(halves).AsInt32()).Store(dst + i);
                        Widen(Vector256.WidenUpper(halves).AsInt32()).Store(dst + i + Vector256<int>.Count);
                    }
                }

                if (Vector128.IsHardwareAccelerated)
                {
                    // 8 halves per iteration (SSE, ARM NEON), and the remainder of the 256-bit loop.
                    for (; i <= length - Vector128<ushort>.Count; i += Vector128<ushort>.Count)
                    {
                        var halves = Vector128.Load(src + i);
                        Widen(Vector128.WidenLower(halves).AsInt32()).Store(dst + i);
                        Widen(Vector128.WidenUpper(halves).AsInt32()).Store(dst + i + Vector128<int>.Count);
                    }
                }

                // Scalar tail, and full fallback without SIMD.
                for (; i < length; i++)
                    dst[i] = (float)halfSource[i];
            }
        }

        /// <summary>
        /// Widens 8 half-precision bit patterns (one per 32-bit lane) to single precision.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> Widen(Vector256<int> half)
        {
            // Move exponent and mantissa into float position and re-bias the exponent. Normal numbers are done here.
            var bits = Vector256.ShiftLeft(half & Vector256.Create(EXPONENT_AND_MANTISSA_MASK), 13);
            var exponent = bits & Vector256.Create(SHIFTED_EXPONENT_MASK);
            bits += Vector256.Create(EXPONENT_REBIAS);

            // Infinity and NaN: push the exponent to all ones, and mark NaNs as quiet, as the hardware conversion does.
            var isInfOrNaN = Vector256.Equals(exponent, Vector256.Create(SHIFTED_EXPONENT_MASK));
            bits += isInfOrNaN & Vector256.Create(INF_OR_NAN_REBIAS);
            var isNaN = isInfOrNaN & ~Vector256.Equals(half & Vector256.Create(MANTISSA_MASK), Vector256<int>.Zero);
            bits |= isNaN & Vector256.Create(QUIET_NAN_BIT);

            // Zero and subnormals: treat the value as 2^-14 * 1.mantissa and subtract 2^-14. The float subtraction
            // renormalizes the result exactly, with no loop or branch.
            var isZeroOrSubnormal = Vector256.Equals(exponent, Vector256<int>.Zero);
            var subnormal = (bits + Vector256.Create(1 << 23)).AsSingle() - Vector256.Create(SUBNORMAL_MAGIC).AsSingle();
            var result = Vector256.ConditionalSelect(isZeroOrSubnormal.AsSingle(), subnormal, bits.AsSingle());

            return result | Vector256.ShiftLeft(half & Vector256.Create(SIGN_MASK), 16).AsSingle();
        }

        /// <summary>
        /// Widens 4 half-precision bit patterns (one per 32-bit lane) to single precision. Same steps as the 256-bit version.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> Widen(Vector128<int> half)
        {
            var bits = Vector128.ShiftLeft(half & Vector128.Create(EXPONENT_AND_MANTISSA_MASK), 13);
            var exponent = bits & Vector128.Create(SHIFTED_EXPONENT_MASK);
            bits += Vector128.Create(EXPONENT_REBIAS);

            var isInfOrNaN = Vector128.Equals(exponent, Vector128.Create(SHIFTED_EXPONENT_MASK));
            bits += isInfOrNaN & Vector128.Create(INF_OR_NAN_REBIAS);
            var isNaN = isInfOrNaN & ~Vector128.Equals(half & Vector128.Create(MANTISSA_MASK), Vector128<int>.Zero);
            bits |= isNaN & Vector128.Create(QUIET_NAN_BIT);

            var isZeroOrSubnormal = Vector128.Equals(exponent, Vector128<int>.Zero);
            var subnormal = (bits + Vector128.Create(1 << 23)).AsSingle() - Vector128.Create(SUBNORMAL_MAGIC).AsSingle();
            var result = Vector128.ConditionalSelect(isZeroOrSubnormal.AsSingle(), subnormal, bits.AsSingle());

            return result | Vector128.ShiftLeft(half & Vector128.Create(SIGN_MASK), 16).AsSingle();
        }
    }
}
