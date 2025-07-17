// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Core
{
    public record CpuExecutionProvider : IExecutionProvider;

    public record CudaExecutionProvider(int GpuId = 0, bool PrimeGpu = false) : IExecutionProvider;

    public record TensorRTExecutionProvider(
        int GpuId = 0,
        TensorRTPrecision Precision = TensorRTPrecision.FP32,
        string EngineCachePath = "")
        : IExecutionProvider;
}
