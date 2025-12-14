namespace YoloDotNet.ExecutionProvider.Cuda.TensorRT
{
    public record TensorRt()
    {
        public TrtPrecision? Precision { get; init; }
        // Precision mode to use: FP32, FP16, or INT8 (requires calibration).

        public int BuilderOptimizationLevel { get; init; } = 3;
        // Optimization level for building the TensorRT engine (0-5).
        // Higher levels spend more time optimizing but increase build time.
        // Levels below 3 may reduce engine performance.

        public string? EngineCachePath { get; init; }
        // Directory path to load/store TensorRT engine and profile cache files.

        public string? EngineCachePrefix { get; init; }
        // Optional prefix for generated engine cache files.
        // Defaults to a standard prefix if not set.

        public string? Int8CalibrationCacheFile { get; init; }
        // Required only for INT8 precision mode.
        // Path to the INT8 calibration cache file used to assign tensor dynamic ranges.
        // Must be generated beforehand via calibration (see demo notes).
    }
}
