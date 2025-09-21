// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.Cuda
{
    public class CudaExecutionProvider : ICuda, IExecutionProvider, IDisposable
    {
        public OnnxDataRecord OnnxData { get; private set; } = default!;

        private InferenceSession _session = default!;
        private OrtIoBinding _ortIoBinding = default!;
        private RunOptions _runOptions = default!;

        private long[] _inputShape = default!;

        /// <summary>
        /// Constructs a CudaExecutionProvider for running ONNX models using CUDA and optionally TensorRT.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="gpuId"></param>
        /// <param name="trtConfig"></param>
        public CudaExecutionProvider(string model, int gpuId = 0, TensorRt? trtConfig = null)
        {
            InitializeYolo(model, gpuId, trtConfig);
        }

        /// <summary>
        /// Overload: Constructs a CudaExecutionProvider for running ONNX models using CUDA and optionally TensorRT.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="gpuId"></param>
        /// <param name="trtConfig"></param>
        public CudaExecutionProvider(byte[] model, int gpuId = 0, TensorRt? trtConfig = null)
        {
            InitializeYolo(model, gpuId, trtConfig);
        }

        /// <summary>
        /// Runs inference on the provided normalized pixel data and returns the inference result.
        /// </summary>
        /// <param name="normalizedPixels"></param>
        /// <param name="tensorBufferSize"></param>
        /// <returns></returns>
        public InferenceResult Run(float[] normalizedPixels, int tensorBufferSize)
        {
            // Deconstruct the input shape into batch size, number of channels, height, and width.
            var (batchSize, colorChannels, height, width) = (OnnxData.InputShape[0], OnnxData.InputShape[1], OnnxData.InputShape[2], OnnxData.InputShape[3]);

            // Create a DenseTensor from the normalized pixel data with the specified shape.
            var tensorPixels = new DenseTensor<float>(
                normalizedPixels.AsMemory(0, tensorBufferSize),
                [batchSize, colorChannels, height, width]
            );

            // Create an OrtValue from the DenseTensor for input to the ONNX model.
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                tensorPixels.Buffer,
                _inputShape
                );

            // Run inference
            using var result = _session.Run(
                _runOptions,
                [OnnxData.InputName],
                [inputOrtValue],
                OnnxData.OutputNames);

            // Extract tensor data from the result
            var tensorData0 = result[0].GetTensorDataAsSpan<float>();
            var tensorData1 = ReadOnlySpan<float>.Empty;

            if (result.Count == 2)
                tensorData1 = result[1].GetTensorDataAsSpan<float>();

            // Return the inference result containing the output tensor data
            return new InferenceResult(tensorData0, tensorData1);
        }

        #region CUDA and TensorRT helper methods

        private void InitializeYolo(object model, int gpuId, TensorRt? trtConfig)
        {
            ConfigureOrtEnv();

            var options = CreateSessionOptions(gpuId, trtConfig);

            // Create session using bytes if available; else load from file with selected provider.
            _session = (model is byte[] modelBytes)
                ? new InferenceSession(modelBytes, options)
                : new InferenceSession((string)model, options);

            _runOptions = new RunOptions();
            _ortIoBinding = _session.CreateIoBinding();
            _session.AllocateGpuMemory(_ortIoBinding, _runOptions);

            GetOnnxMetaData();

            // Set the input shape for creating tensors during inference.
            _inputShape = [.. OnnxData.InputShape.Select(i => (long)i)];
        }

        private SessionOptions CreateSessionOptions(int gpuId, TensorRt? trtConfig)
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            if (gpuId >= 0)
            {
                if (trtConfig is not null)
                {
                    options.ConfigureTensorRT(gpuId, trtConfig);
                }
                else
                {
                    ConfigureCuda(gpuId, options);
                }
            }
            else if (gpuId == -1)
            {
                options.EnableCpuMemArena = true;
            }
            else
            {
                throw new ArgumentOutOfRangeException(
                    paramName: nameof(gpuId),
                    actualValue: gpuId,
                    message: "The specified gpuId is not valid. Use -1 for CPU execution, or 0 and above for a GPU device ID.");
            }
            
            return options;
        }

        private static void ConfigureOrtEnv()
        {
            try
            {
                // Log errors and fatals
                var envOptions = new EnvironmentCreationOptions
                {
                    logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
                };

                OrtEnv.CreateInstanceWithOptions(ref envOptions);
            }
            catch (OnnxRuntimeException ex) when (ex.Message.Contains("OrtEnv singleton instance already exists"))
            {
                // OrtEnv has already been initialized — ignore and continue gracefully...
            }
        }

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

        private void GetOnnxMetaData()
        {
            var metaData = _session.ModelMetadata.CustomMetadataMap;

            OnnxData = new OnnxDataRecord(
                metaData,
                _session.InputNames[0],
                [.. _session.OutputNames],
                _session.InputMetadata.Values.Select(x => x.Dimensions).FirstOrDefault() ?? [],
                [.. _session.OutputMetadata.Values.Select(x => x.Dimensions)],
                metaData["names"]
            );
        }

        public void Dispose()
        {
            _session?.Dispose();
            _runOptions?.Dispose();
            _ortIoBinding?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
