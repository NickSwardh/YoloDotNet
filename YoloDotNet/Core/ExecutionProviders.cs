// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Core
{
    public record CpuExecutionProvider : IExecutionProvider;

    public record CudaExecutionProvider(int GpuId = 0, bool PrimeGpu = false) : IExecutionProvider;

    public record TensorRtExecutionProvider() : ITensorRTExecutionProvider
    {
        public TrtPrecision Precision { get; init; }
        // Precision mode to use: FP32, FP16, or INT8 (requires calibration).

        public int GpuId { get; init; }
        // GPU device index to run TensorRT operations on.

        public int BuilderOptimizationLevel { get; init; } = 3;
        // Optimization level for building the TensorRT engine (0-5).
        // Higher levels spend more time optimizing but increase build time.
        // Levels below 3 may reduce engine performance.

        public string? EngineCachePath { get; init; }
        // Directory path to load/store TensorRT engine and profile cache files.

        public string? Int8CalibrationCacheFile { get; init; }
        // Required only for INT8 precision mode.
        // Path to the INT8 calibration cache file used to assign tensor dynamic ranges.
        // Must be generated beforehand via calibration (see demo notes).

        public string? EngineCachePrefix { get; init; }
        // Optional prefix for generated engine cache files.
        // Defaults to a standard prefix if not set.
    }
}
