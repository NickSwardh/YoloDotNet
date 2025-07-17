// SPDX-License-Identifier: GPL-3.0-or-later
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
                EnableCpuMemArena = true
            };

            switch (config)
            {
                case CpuExecutionProvider :
                    ConfigureCpu(options);
                    break;

                case CudaExecutionProvider cudaProvider :
                    ConfigureCuda(cudaProvider, options);
                    break;

                case TensorRTExecutionProvider tensorProvider :
                    ConfigureTensorRT(tensorProvider, options);
                    break;

                default:
                    throw new ArgumentException($"Unknown execution provider: {config.GetType().Name}");
            }

            return options;
        }

        private static void ConfigureCpu(SessionOptions options)
        {
            options.EnableCpuMemArena = true;
        }

        private static void ConfigureCuda(CudaExecutionProvider provider, SessionOptions options)
        {
            var cudaOptions = new OrtCUDAProviderOptions();

            cudaOptions.UpdateOptions(new Dictionary<string, string>
            {
                { "device_id", provider.GpuId.ToString() },  
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

        private static void ConfigureTensorRT(TensorRTExecutionProvider provider, SessionOptions options)
        {
            var tensorOptions = new OrtTensorRTProviderOptions();

            var providerOptions = new Dictionary<string, string>
            {
                { "device_id", provider.GpuId.ToString()},
                // Specifies which GPU device to use (default = 0 if not set).

                { "trt_engine_cache_enable", "1" },
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

                { "trt_engine_cache_path", provider.EngineCachePath },
                // Specify path for TensorRT engine and profile files.
                    
                { "trt_engine_cache_prefix", "YoloDotNet" },
                // Set prefix. If empty a default prefix will be used.

                { "trt_auxiliary_streams", "0" },
                // Set maximum number of auxiliary streams per inference stream. 0 = optimal memory usage.
            };

            switch (provider.Precision)
            {
                case TensorRTPrecision.FP16:
                    providerOptions.Add("trt_fp16_enable", "1");
                    // Enable FP16 mode in TensorRT.
                    // Note: not all Nvidia GPUs support FP16 precision!
                    break;
                case TensorRTPrecision.INT8:
                    providerOptions.Add("trt_int8_enable", "1");
                    // Enable INT8 mode in TensorRT.
                    // Note: not all Nvidia GPUs support INT8 precision!
                    break;
            }

            tensorOptions.UpdateOptions(providerOptions);
            options.AppendExecutionProvider_Tensorrt(tensorOptions);
        }
    }
}
