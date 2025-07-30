﻿// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Core
{
    internal class ExecutionProviderFactory
    {
        public static SessionOptions Create(IExecutionProvider config)
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            };

            switch (config)
            {
                case CpuExecutionProvider:
                    ConfigureCpu(options);
                    break;

                case CudaExecutionProvider cudaProvider:
                    ConfigureCuda(cudaProvider.GpuId, options);
                    break;

                case TensorRtExecutionProvider trtProvider:
                    ConfigureTensorRT(trtProvider, options);
                    break;
                default:
                    throw new YoloDotNetUnsupportedProviderException($"Unknown execution provider: {config.GetType().Name}");
            }

            return options;
        }

        private static void ConfigureCpu(SessionOptions options)
            => options.EnableCpuMemArena = true;

        private static void ConfigureCuda(int gpuId, SessionOptions options)
        {
            var cudaOptions = new OrtCUDAProviderOptions();

            cudaOptions.UpdateOptions(new Dictionary<string, string>
            {
                { "device_id", gpuId.ToString() },
                // Specifies which GPU device to use (default = 0 if not set).

                { "arena_extend_strategy", "kNextPowerOfTwo" }, 
                // Controls how the GPU memory arena grows when more memory is needed.
                // kNextPowerOfTwo doubles the allocation size to the next power of two,
                // which reduces the frequency of CUDA malloc/free calls and minimizes fragmentation 
                // in long-running or high-throughput inference scenarios like YOLO object detection.

                { "cudnn_conv_algo_search", "EXHAUSTIVE" },
                // Forces cuDNN to benchmark all available convolution algorithms during model initialization
                // and select the fastest one for the hardware + model combination.
                // This gives optimal conv kernel performance at runtime, especially beneficial for large or custom conv layers.
            });

            options.AppendExecutionProvider_CUDA(cudaOptions);
        }

        private static void ConfigureTensorRT(ITensorRTExecutionProvider provider, SessionOptions options)
        {
            var engineCache = provider.EngineCachePath;
            var cacheEnabled = 1;

            // If a engine cache is used, verify the path exist.
            if (string.IsNullOrEmpty(engineCache) is false && Directory.Exists(engineCache) is false)
            {
                throw new YoloDotNetExecutionProviderException($"TensorRT engine cache directory not found: '{engineCache}'. " +
                    "This folder is required to store and load TensorRT engine cache. " +
                    "Make sure the path exists and is accessible by the application. " +
                    "You can configure it using the 'EngineCachePath' parameter of the execution provider.");
            }
            else if (string.IsNullOrEmpty(engineCache) is true)
            {
                cacheEnabled = 0;
            }
            
            var tensorOptions = new OrtTensorRTProviderOptions();

            var providerOptions = new Dictionary<string, string>
            {
                { "device_id", provider.GpuId.ToString()},
                // Specifies which GPU device to use (default = 0 if not set).

                { "trt_engine_cache_enable", cacheEnabled.ToString() },
                // Enable caching.
                // Engine will be cached when it’s built for the first time so next time
                // when new inference session is created the engine can be loaded directly
                // from cache.In order to validate that the loaded engine is usable for
                // current inference, engine profile is also cached and loaded along with
                // engine.If current input shapes are in the range of the engine profile,
                // the loaded engine can be safely used.Otherwise if input shapes are out
                // of range, profile cache will be updated to cover the new shape and
                // engine will be recreated based on the new profile (and also refreshed
                // in the engine cache).
                //
                // Note:
                // each engine is created for specific settings such as model path/name,
                // precision (FP32/FP16/INT8 etc), workspace, profiles etc, and specific
                // GPUs and it’s not portable, so it’s essential to make sure those
                // settings are not changing, otherwise the engine needs to be rebuilt
                // and cached again.

                { "trt_engine_cache_path", engineCache ?? Directory.GetCurrentDirectory() },
                // Specify path for TensorRT engine and profile files.
                // If no path is specifiec, executable folder will be used.


                { "trt_engine_cache_prefix", provider.EngineCachePrefix ?? "YoloDotNet" },
                // Set prefix. If empty a default prefix will be used.

                { "trt_auxiliary_streams", "0" },
                // Set maximum number of auxiliary streams per inference stream. 0 = optimal memory usage.

                { "trt_builder_optimization_level", provider.BuilderOptimizationLevel.ToString() }
                // Set the builder optimization level to use when building the engine. A higher level
                // allows TensorRT to spend more building time on more optimization options.
                //
                // WARNING: levels below 3 do not guarantee good engine performance, but greatly improve
                // build time. Default 3, valid range[0 - 5].
            };
            
            switch (provider.Precision)
            {
                case TrtPrecision.FP16:
                    providerOptions.Add("trt_fp16_enable", "1");
                    // Enable FP16 mode in TensorRT.
                    // Note: not all Nvidia GPUs support FP16 precision!

                    break;
                case TrtPrecision.INT8:

                    var calibrationCache = provider.Int8CalibrationCacheFile;

                    if (File.Exists(calibrationCache) is false)
                        throw new YoloDotNetExecutionProviderException($"INT8 calibration cache file not found: '{calibrationCache}'. " +
                            "This file is required to initialize the model in INT8 precision mode using TensorRT. " +
                            "Make sure the file exists and is accessible by the application. " +
                            $"You can specify its location using the '{nameof(provider.Int8CalibrationCacheFile)}' parameter of the execution provider.");

                    providerOptions.Add("trt_int8_enable", "1");
                    // Enable INT8 mode in TensorRT.
                    // Note: not all Nvidia GPUs support INT8 precision!

                    providerOptions.Add("trt_int8_use_native_calibration_table", "1");
                    // Select what calibration table is used for non - quantized models in INT8 mode.
                    // The calibration table is specific to models and calibration data sets.
                    // Whenever new calibration table is generated, old file in the path should
                    // be cleaned up or be replaced.
                    //
                    // Note: Old calibration table cache files are not removed!

                    providerOptions.Add("trt_int8_calibration_table_name", calibrationCache);
                    // Specifies the path to the INT8 calibration table file used during engine building.
                    // Required for non-QDQ models running in INT8 mode.
                    // TensorRT uses this table to assign dynamic ranges to tensors.
                    // The table must be pre-generated using calibration with representative data.

                    // To generate the calibration cache, export the model to TensorRT using the Ultralytics library:
                    //
                    //   yolo export model=yolo11s.pt format=engine int8=true simplify=true opset=17
                    //
                    // This command will generate:
                    //   - An ONNX model (unquantized, FP32-based)
                    //   - A TensorRT engine optimized for INT8 precision
                    //   - A calibration *.cache file used by TensorRT for INT8 execution
                    //
                    // The path to the *.cache file is required in order to run YOLO ONNX models in INT8 mixed precision mode in .NET.

                    break;
            }
            
            tensorOptions.UpdateOptions(providerOptions);
            options.AppendExecutionProvider_Tensorrt(tensorOptions);
        }
    }
}
