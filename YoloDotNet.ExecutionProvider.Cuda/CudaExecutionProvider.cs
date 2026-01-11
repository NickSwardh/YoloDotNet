// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.Cuda
{
    public class CudaExecutionProvider : IExecutionProvider, IDisposable
    {
        public OnnxDataRecord OnnxData { get; private set; } = default!;

        #region Fields
        private InferenceSession _session = default!;
        private OrtIoBinding _ortIoBinding = default!;
        private RunOptions _runOptions = default!;

        private float[] _outputBuffer0 = default!;
        private float[] _outputBuffer1 = default!;

        private long[] _inputShape = default!;
        private int _inputShapeSize;
        private TensorElementType _elementDataType = default!;
        private int _dataTypeSize;
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

            _runOptions = new RunOptions();
            _ortIoBinding = _session.CreateIoBinding();
            _session.AllocateGpuMemory(_ortIoBinding, _runOptions, _elementDataType);

            // Set the input shape for creating tensors during inference.
            _inputShape = [.. OnnxData.InputShape.Select(i => (long)i)];

        }
        #endregion

        #region Run Inference
        /// <summary>
        /// Runs inference on the provided normalized pixel data.
        /// </summary>
        /// <param name="normalizedPixels"></param>
        /// <returns></returns>
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

                // Run inference
                using var result = _session.Run(
                    _runOptions,
                    [OnnxData.InputName],
                    [inputOrtValue],
                    OnnxData.OutputNames);

                // Handle output based on the model's data type
                if (_elementDataType == TensorElementType.Float)
                {
                    // Extract tensor data from the result
                    var tensorData0 = result[0].GetTensorDataAsSpan<float>();
                    var tensorData1 = ReadOnlySpan<float>.Empty;

                    if (result.Count == 2)
                        tensorData1 = result[1].GetTensorDataAsSpan<float>();

                    // Return the inference result containing the output tensor data
                    return new InferenceResult(tensorData0, tensorData1);
                }
                else
                {
                    var tensorData0 = result[0].GetTensorDataAsSpan<Float16>();
                    var tensorData1 = ReadOnlySpan<Float16>.Empty;

                    if (result.Count == 2)
                        tensorData1 = result[1].GetTensorDataAsSpan<Float16>();

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

            int items;
            int elements;

            // Calculate the total number of elements for each output tensor and allocate buffers.

            // Classification models only has one output tensor with shape [1, num_classes]
            if (OnnxData.OutputShapes[0].Length == 2)
            {
                (items, elements) = (OnnxData.OutputShapes[0][0], OnnxData.OutputShapes[0][1]);
                _outputBuffer0 = new float[elements * items];
            }
            // All other models has an output tensor with shape [1, num_boxes, num_attributes]
            else
            {
                (items, elements) = (OnnxData.OutputShapes[0][1], OnnxData.OutputShapes[0][2]);
                _outputBuffer0 = new float[elements * items];
            }

            // If there is a second output tensor (segmentation), allocate a buffer for it as well.

            // If there is a second output tensor, allocate a buffer for it as well.
            if (OnnxData.OutputShapes.Length == 2)
            {
                (items, elements) = (OnnxData.OutputShapes[1][1], OnnxData.OutputShapes[1][2]);
                _outputBuffer1 = new float[elements * items];
            }
        }

        /// <summary>
        /// Creates and configures session options for the ONNX Runtime session.
        /// </summary>
        /// <param name="gpuId"></param>
        /// <param name="trtConfig"></param>
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
        /// <param name="gpuId"></param>
        /// <param name="options"></param>
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
        /// <param name="tensorData0"></param>
        /// <param name="tensorData1"></param>
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
        {
            // Extract custom metadata from the ONNX model.
            var metaData = _session.ModelMetadata.CustomMetadataMap;

            // Get input shape and size.
            var inputShape = Array.ConvertAll(_session.InputMetadata[_session.InputNames[0]].Dimensions, Convert.ToInt64);

            _inputShapeSize = (int)ShapeUtils.GetSizeForShape(inputShape);
            _elementDataType = GetModelElementType();
            _dataTypeSize = _elementDataType == TensorElementType.Float16 ? sizeof(ushort) : sizeof(float);

            // Determine model data type (Float32 or Float16).
            var modelDataType = _elementDataType == TensorElementType.Float16
                ? ModelDataType.Float16
                : ModelDataType.Float;

            // Create OnnxDataRecord to hold model information.
            OnnxData = new OnnxDataRecord(
                metaData,
                modelDataType,
                _session.InputNames[0],
                [.. _session.OutputNames],
                _session.InputMetadata.Values.Select(x => x.Dimensions).First(),
                [.. _session.OutputMetadata.Values.Select(x => x.Dimensions)],
                _inputShapeSize,
                metaData["names"]
            );
        }

        /// <summary>
        /// Gets the tensor element type used by the model (e.g., Float32 or Float16).
        /// </summary>
        internal TensorElementType GetModelElementType()
            => _session.InputMetadata["images"].ElementDataType;

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