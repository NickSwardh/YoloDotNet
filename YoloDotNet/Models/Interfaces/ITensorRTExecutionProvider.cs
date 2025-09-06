// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models.Interfaces
{
    public interface ITensorRTExecutionProvider : IExecutionProvider
    {
        TrtPrecision Precision { get; }
        int GpuId { get; }
        int BuilderOptimizationLevel { get; }
        string? EngineCachePath { get; }
        string? Int8CalibrationCacheFile { get; }
        string? EngineCachePrefix { get; }
    }
}
