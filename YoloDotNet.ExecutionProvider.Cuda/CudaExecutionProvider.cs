// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.Cuda
{
    public class CudaExecutionProvider : IExecutionProvider, IDisposable
    {
        public OnnxModel OnnxData { get; private set; } = default!;
        public object Session => _session;

        #region Private Fields
        private InferenceSession _session = default!;
        private OrtIoBinding _ortIoBinding = default!;
        private RunOptions _runOptions = default!;

        private float[] _outputBuffer0 = default!;
        private float[] _outputBuffer1 = default!;

        private long[] _inputShape = default!;
        private string[] _inputNames = default!;
        private int _inputShapeSize;
        private List<string> _outputNames = default!;
        private int _dataTypeSize;
        private TensorElementType _elementDataType = default!;

        private IDisposableReadOnlyCollection<OrtValue>? _currentResult;

        // store unmanaged allocation so we can free it later
        private nint _gpuAllocPtr = nint.Zero;

        #endregion

        #region Constructors
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
        #endregion

        #region Initialization
        /// <summary>
        /// Initializes the ONNX Runtime session, configures the CUDA execution provider and allocates resources.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="gpuId"></param>
        /// <param name="trtConfig"></param>
        private void InitializeYolo(object model, int gpuId, TensorRt? trtConfig)
        {
            ConfigureOrtEnv();

            var options = CreateSessionOptions(gpuId, trtConfig);

            // Create session using bytes if available; else load from file with selected provider.
            _session = (model is byte[] modelBytes)
                ? new InferenceSession(modelBytes, options)
                : new InferenceSession((string)model, options);

            GetOnnxMetaData();
            AllocateOutputBuffers();
            InitializeInferenceParameters();

            _runOptions = new RunOptions();
            _ortIoBinding = _session.CreateIoBinding();
            _gpuAllocPtr = _session.AllocateGpuMemory(_ortIoBinding, _runOptions, _elementDataType);

        }

        private void InitializeInferenceParameters()
        {
            var firstInput = OnnxData.InputShapes.FirstOrDefault();

            if (EqualityComparer<KeyValuePair<string, long[]>>.Default.Equals(firstInput, default))
                throw new YoloDotNetException("Corrupt or incompatible model. No input shape was found.");

            _inputNames = [firstInput.Key];
            _inputShape = firstInput.Value;
            _inputShapeSize = OnnxData.InputShapeSize;
            _outputNames = [.. OnnxData.OutputShapes.Select(x => x.Key)];

            if (OnnxData.ModelDataType == ModelDataType.Float)
            {
                _elementDataType = TensorElementType.Float;
                _dataTypeSize = sizeof(float);
            }
            else
            {
                _elementDataType = TensorElementType.Float16;
                _dataTypeSize = sizeof(ushort);
            }
        }

        #endregion

        #region Run Inference
        /// <summary>
        /// Runs inference on the provided normalized pixel data.
        /// </summary>
        /// <param name="normalizedPixels"></param>
        unsafe public InferenceResult Run<T>(T[] normalizedPixels) where T : unmanaged
        {
            // Pin the input pixel data in memory to prevent it from being moved by the garbage collector.
            fixed (T* pData = normalizedPixels)
            {
                // Create an OrtValue tensor from the pinned data
                using var inputOrtValue = OrtValue.CreateTensorValueWithData(
                    OrtMemoryInfo.DefaultInstance,
                    _elementDataType,
                    _inputShape,
                    (IntPtr)pData,
                    _inputShapeSize * _dataTypeSize // size in bytes (ushort is 2 bytes, float is 4 bytes)
                );

                // Dispose previous result before running new inference to prevent memory leaks
                _currentResult?.Dispose();

                // Run inference
                _currentResult = _session.Run(
                    _runOptions,
                    _inputNames,
                    [inputOrtValue],
                    _outputNames);

                // Handle output based on the model's data type
                if (_elementDataType == TensorElementType.Float)
                {
                    // Extract tensor data from the result
                    var tensorData0 = _currentResult[0].GetTensorDataAsSpan<float>();
                    var tensorData1 = ReadOnlySpan<float>.Empty;

                    if (_currentResult.Count == 2)
                        tensorData1 = _currentResult[1].GetTensorDataAsSpan<float>();

                    // Return the inference result containing the output tensor data
                    return new InferenceResult(tensorData0, tensorData1);
                }
                else
                {
                    var tensorData0 = _currentResult[0].GetTensorDataAsSpan<Float16>();
                    var tensorData1 = ReadOnlySpan<Float16>.Empty;

                    if (_currentResult.Count == 2)
                        tensorData1 = _currentResult[1].GetTensorDataAsSpan<Float16>();

                    ConvertFloat16ToFloat(tensorData0, tensorData1);

                    // Return the inference result containing the output tensor data
                    return new InferenceResult(_outputBuffer0, _outputBuffer1);
                }
            }
        }
        #endregion

        #region CUDA and TensorRT helper methods
        /// <summary>
        /// Allocates float output buffers for models using Float16 data type.
        /// </summary>
        private void AllocateOutputBuffers()
        {
            // Pre-allocate output buffers if model uses Float16 to avoid repeated allocations during inference.
            if (OnnxData.ModelDataType == ModelDataType.Float)
                return;

            int batch, attributes;
            var outputShape = OnnxData.OutputShapes.ElementAt(0).Value;

            if (OnnxData.ModelType == ModelType.Classification)
            {
                // For classification models, output shape is [Batch, Attributes]
                (batch, attributes) = (outputShape[0], outputShape[1]);
            }
            else
            {
                // For other models, output shape is [Batch, Attributes, Predictions]
                (batch, attributes) = (outputShape[1], outputShape[2]);
            }

            _outputBuffer0 = new float[attributes * batch];

            // Allocate second output buffer if model has two outputs (e.g., segmentation models).
            if (OnnxData.OutputShapes.Count == 2)
            {
                outputShape = OnnxData.OutputShapes.ElementAt(1).Value;
                var size = outputShape[1] * outputShape[2];
                _outputBuffer1 = new float[size];
            }
        }

        /// <summary>
        /// Creates and configures session options for the ONNX Runtime session.
        /// </summary>
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

        /// <summary>
        /// Configure the global OrtEnv instance with custom logging options.
        /// </summary>
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

        /// <summary>
        /// Configures the session options to use the CUDA execution provider with specified options.
        /// </summary>
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

                // Reduce host/device synchronization by enabling copy using the default stream when supported.
                // This can avoid implicit stream synchronizations on some ONNX Runtime builds.
                { "do_copy_in_default_stream", "1" },

                // Allow cuDNN to use the maximum workspace (may increase memory usage but can improve kernel perf).
                { "cudnn_conv_use_max_workspace", "1" }

            });

            options.AppendExecutionProvider_CUDA(cudaOptions);
        }

        /// <summary>
        /// Converts Float16 tensor data to Float32 and stores it in pre-allocated output buffers.
        /// </summary>
        unsafe private void ConvertFloat16ToFloat(ReadOnlySpan<Float16> tensorData0, ReadOnlySpan<Float16> tensorData1)
        {
            fixed (Float16* src0 = tensorData0)
            fixed (Float16* src1 = tensorData1)
            fixed (float* dst0 = _outputBuffer0)
            fixed (float* dst1 = _outputBuffer1)
            {
                int len0 = tensorData0.Length;
                int len1 = tensorData1.Length;

                for (int i = 0; i < len0; i++)
                    dst0[i] = (float)src0[i];

                for (int i = 0; i < len1; i++)
                    dst1[i] = (float)src1[i];
            }
        }

        /// <summary>
        /// Extracts metadata and input/output shapes from the ONNX model.
        /// </summary>
        private void GetOnnxMetaData()
            => OnnxData = _session.ParseOnnx();

        public void Dispose()
        {
            // Dispose any current results first.
            _currentResult?.Dispose();

            // Free unmanaged GPU allocation if it was allocated.
            if (_gpuAllocPtr != nint.Zero)
            {
                try
                {
                    Marshal.FreeHGlobal(_gpuAllocPtr);
                }
                catch
                {
                    // swallow — best-effort free; do not throw from Dispose.
                }
                _gpuAllocPtr = nint.Zero;
            }

            _session?.Dispose();
            _runOptions?.Dispose();
            _ortIoBinding?.Dispose();
            _currentResult?.Dispose();

            GC.SuppressFinalize(this);
        }
        #endregion
    }
}