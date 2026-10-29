// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Core
{
    public ref struct InferenceResult
    {
        public ReadOnlySpan<float> OrtSpan0 { get; set; }
        public ReadOnlySpan<float> OrtSpan1 { get; set; }
        public ReadOnlySpan<float> OrtSpan2 { get; set; }
        public SKSizeI ImageOriginalSize { get; set; }

        public InferenceResult()
        {
        }

        public InferenceResult(
            ReadOnlySpan<float> ortSpan0,
            ReadOnlySpan<float> ortSpan1,
            ReadOnlySpan<float> ortSpan2 = default)
        {
            OrtSpan0 = ortSpan0;
            OrtSpan1 = ortSpan1;
            OrtSpan2 = ortSpan2;
        }
    }
}
