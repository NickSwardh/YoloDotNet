// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models.Interfaces
{
    public interface IExecutionProvider
    {
        public InferenceResult Run(float[] normalizedPixels, int tensorBufferSize);
        public OnnxDataRecord OnnxData { get; }
    }
}
