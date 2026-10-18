// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Core
{
    /// <summary>
    /// The output tensors of one inference call.
    /// </summary>
    /// <remarks>
    /// Each call owns its own output memory, so concurrent calls never read or free each other's tensors.
    /// The spans stay valid until <see cref="Dispose"/> is called, which releases the ONNX Runtime output
    /// and returns any rented conversion buffers. Dispose exactly once, after all post-processing is done.
    /// </remarks>
    public ref struct InferenceResult
    {
        public ReadOnlySpan<float> OrtSpan0 { get; set; }
        public ReadOnlySpan<float> OrtSpan1 { get; set; }
        public SKSizeI ImageOriginalSize { get; set; }

        private IDisposable? _ortOutputs;
        private float[]? _rentedBuffer0;
        private float[]? _rentedBuffer1;

        public InferenceResult()
        {
        }

        public InferenceResult(
            ReadOnlySpan<float> ortSpan0,
            ReadOnlySpan<float> ortSpan1)
        {
            OrtSpan0 = ortSpan0;
            OrtSpan1 = ortSpan1;
        }

        /// <summary>
        /// Creates a result that owns its output memory.
        /// </summary>
        /// <param name="ortOutputs">The ONNX Runtime output collection the spans point into, or null if the spans point into rented buffers only.</param>
        /// <param name="ortSpan0">First output tensor.</param>
        /// <param name="ortSpan1">Second output tensor, or empty.</param>
        /// <param name="rentedBuffer0">Buffer rented from <see cref="ArrayPool{T}.Shared"/> that backs <paramref name="ortSpan0"/>, if any.</param>
        /// <param name="rentedBuffer1">Buffer rented from <see cref="ArrayPool{T}.Shared"/> that backs <paramref name="ortSpan1"/>, if any.</param>
        public InferenceResult(
            IDisposable? ortOutputs,
            ReadOnlySpan<float> ortSpan0,
            ReadOnlySpan<float> ortSpan1,
            float[]? rentedBuffer0 = null,
            float[]? rentedBuffer1 = null)
        {
            _ortOutputs = ortOutputs;
            _rentedBuffer0 = rentedBuffer0;
            _rentedBuffer1 = rentedBuffer1;
            OrtSpan0 = ortSpan0;
            OrtSpan1 = ortSpan1;
        }

        /// <summary>
        /// Releases the output memory owned by this result. The spans must not be used afterwards.
        /// </summary>
        public void Dispose()
        {
            OrtSpan0 = default;
            OrtSpan1 = default;

            _ortOutputs?.Dispose();
            _ortOutputs = null;

            if (_rentedBuffer0 is not null)
                ArrayPool<float>.Shared.Return(_rentedBuffer0);

            if (_rentedBuffer1 is not null)
                ArrayPool<float>.Shared.Return(_rentedBuffer1);

            _rentedBuffer0 = null;
            _rentedBuffer1 = null;
        }
    }
}
