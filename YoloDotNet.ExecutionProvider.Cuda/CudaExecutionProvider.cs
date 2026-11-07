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
        private CudaGraphRunner? _cudaGraph;
        private readonly object _runLock = new();

        private long[] _inputShape = default!;
        private string[] _inputNames = default!;
        private int _inputShapeSize;
        private List<string> _outputNames = default!;
        private int _dataTypeSize;
        private TensorElementType _elementDataType = default!;

        // store unmanaged allocation so we can free it later
        private nint _gpuAllocPtr = nint.Zero;

        #endregion

        #region Constructors
        /// <summary>
        /// Constructs a CudaExecutionProvider for running ONNX models using CUDA and optionally TensorRT.
        /// </summary>
        /// <param name="model">Path to the ONNX model file.</param>
        /// <param name="gpuId">GPU device ID (0 and above), or -1 for CPU execution.</param>
        /// <param name="trtConfig">TensorRT settings. When set, the model runs on TensorRT with CUDA as fallback.</param>
        /// <param name="enableCudaGraph">
        /// Records the GPU work on the first inference and replays it on later calls, for CUDA and TensorRT alike.
        /// Requires a GPU device ID of 0 or above and a model that runs entirely on the GPU
        /// (entirely on TensorRT when <paramref name="trtConfig"/> is set). Models that don't qualify fail when the
        /// provider is created.
        /// </param>
        public CudaExecutionProvider(string model, int gpuId = 0, TensorRt? trtConfig = null, bool enableCudaGraph = false)
        {
            InitializeYolo(model, gpuId, trtConfig, enableCudaGraph);
        }

        /// <inheritdoc cref="CudaExecutionProvider(string, int, TensorRt?, bool)"/>
        /// <param name="model">The ONNX model as a byte array.</param>
        public CudaExecutionProvider(byte[] model, int gpuId = 0, TensorRt? trtConfig = null, bool enableCudaGraph = false)
        {
            InitializeYolo(model, gpuId, trtConfig, enableCudaGraph);
        }
        #endregion

        #region Initialization
        /// <summary>
        /// Initializes the ONNX Runtime session, configures the CUDA execution provider and allocates resources.
        /// </summary>
        /// <param name="model">Model file path or bytes.</param>
        /// <param name="gpuId">GPU device ID, or -1 for CPU.</param>
        /// <param name="trtConfig">Optional TensorRT settings.</param>
        /// <param name="enableCudaGraph">Replay the inference as a CUDA graph (CUDA or TensorRT).</param>
        private void InitializeYolo(object model, int gpuId, TensorRt? trtConfig, bool enableCudaGraph)
        {
            if (enableCudaGraph && gpuId < 0)
                throw new ArgumentException("CUDA Graphs require a GPU device ID of 0 or greater.", nameof(gpuId));

            ConfigureOrtEnv();

            using var options = CreateSessionOptions(gpuId, trtConfig, enableCudaGraph);

            // Create session using bytes if available; else load from file with selected provider.
            _session = (model is byte[] modelBytes)
                ? new InferenceSession(modelBytes, options)
                : new InferenceSession((string)model, options);

            try
            {
                GetOnnxMetaData();
                InitializeInferenceParameters();

                _runOptions = new RunOptions();
                if (enableCudaGraph)
                    _cudaGraph = new CudaGraphRunner(_session, gpuId);
                else
                {
                    _ortIoBinding = _session.CreateIoBinding();
                    _gpuAllocPtr = _session.AllocateGpuMemory(_ortIoBinding, _runOptions, _elementDataType);
                }
            }
            catch
            {
                Dispose();
                throw;
            }

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
            if (_cudaGraph is not null)
            {
                // Device addresses are shared by graph replays. Upload, inference and output snapshots must
                // stay inside the same lock; the returned host buffers belong exclusively to this call.
                lock (_runLock)
                    return _cudaGraph.Run(normalizedPixels, _runOptions);
            }

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

                // Run inference. The output belongs to this call only: it travels with the returned InferenceResult
                // and is released when the caller disposes it, so concurrent calls never share or free each other's output.
                IDisposableReadOnlyCollection<OrtValue> outputs;

                // Concurrent Run calls on one CUDA session returned other images' results and failed with CUDA errors,
                // so inference on this session is serialized. Only the inference call is locked: resizing, normalization
                // and post-processing of other calls still run in parallel with it.
                lock (_runLock)
                {
                    outputs = _session.Run(
                        _runOptions,
                        _inputNames,
                        [inputOrtValue],
                        _outputNames);
                }

                return CreateInferenceResult(outputs);
            }
        }
        #endregion

        #region CUDA and TensorRT helper methods

        /// <summary>
        /// Creates and configures session options for the ONNX Runtime session.
        /// </summary>
        private SessionOptions CreateSessionOptions(int gpuId, TensorRt? trtConfig, bool enableCudaGraph)
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
                    options.ConfigureTensorRT(gpuId, trtConfig, enableCudaGraph);
                }

                // Keep operations TensorRT cannot execute on CUDA where possible, before falling back to CPU.
                // With TensorRT, the graph is captured by TensorRT, so CUDA keeps graph capture off.
                ConfigureCuda(gpuId, options, enableCudaGraph && trtConfig is null);
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
        private static void ConfigureCuda(int gpuId, SessionOptions options, bool enableCudaGraph)
        {
            using var cudaOptions = new OrtCUDAProviderOptions();

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
                { "cudnn_conv_use_max_workspace", "1" },

                { "enable_cuda_graph", enableCudaGraph ? "1" : "0" }

            });

            options.AppendExecutionProvider_CUDA(cudaOptions);
        }

        /// <summary>
        /// Wraps the output tensors of one inference call in an <see cref="InferenceResult"/> that owns them.
        /// </summary>
        /// <remarks>
        /// Each output is read by its own tensor type, not the input's: an FP16 model can return FP32 outputs.
        /// Float outputs are handed over as they are; the caller releases them by disposing the result.
        /// Float16 outputs are converted into float buffers rented for this call only, sized from the actual
        /// tensors. When no span points into the ONNX Runtime outputs, they are released right away.
        /// Either way no two calls share output memory, so concurrent calls cannot read or free each other's tensors.
        /// </remarks>
        private static InferenceResult CreateInferenceResult(IDisposableReadOnlyCollection<OrtValue> outputs)
        {
            float[]? buffer0 = null;
            float[]? buffer1 = null;
            ReadOnlySpan<float> tensorData0;
            ReadOnlySpan<float> tensorData1;

            try
            {
                tensorData0 = ReadOutput(outputs[0], out buffer0);
                tensorData1 = outputs.Count == 2 ? ReadOutput(outputs[1], out buffer1) : ReadOnlySpan<float>.Empty;
            }
            catch
            {
                outputs.Dispose();

                if (buffer0 is not null)
                    ArrayPool<float>.Shared.Return(buffer0);

                if (buffer1 is not null)
                    ArrayPool<float>.Shared.Return(buffer1);

                throw;
            }

            var keepOutputs = buffer0 is null || (outputs.Count == 2 && buffer1 is null);

            if (keepOutputs is false)
                outputs.Dispose();

            return new InferenceResult(keepOutputs ? outputs : null, tensorData0, tensorData1, buffer0, buffer1);
        }

        /// <summary>
        /// Returns the output as floats. Float16 output is converted into a buffer rented from <see cref="ArrayPool{T}.Shared"/>.
        /// </summary>
        private static ReadOnlySpan<float> ReadOutput(OrtValue output, out float[]? rentedBuffer)
        {
            if (output.GetTensorTypeAndShape().ElementDataType == TensorElementType.Float)
            {
                rentedBuffer = null;
                return output.GetTensorDataAsSpan<float>();
            }

            var halfData = output.GetTensorDataAsSpan<Float16>();
            rentedBuffer = ArrayPool<float>.Shared.Rent(halfData.Length);
            ConvertFloat16ToFloat(halfData, rentedBuffer);

            return rentedBuffer.AsSpan(0, halfData.Length);
        }

        /// <summary>
        /// Converts Float16 tensor data to Float32.
        /// </summary>
        /// <remarks>
        /// ONNX Runtime's <see cref="Float16"/> and .NET's <see cref="Half"/> share the IEEE 754 binary16 layout, so the
        /// data is reinterpreted as <see cref="Half"/> and converted with the vectorized
        /// <see cref="Float16Extensions.ConvertToSingle"/>. ONNX Runtime's own <see cref="Float16"/> cast gives the same
        /// values but is slow for subnormal values, which dominate model outputs (class scores close to zero): about
        /// 4.1 ms for a 640x640 detection output, against about 0.14 ms here.
        /// </remarks>
        private static void ConvertFloat16ToFloat(ReadOnlySpan<Float16> source, float[] destination)
            => MemoryMarshal.Cast<Float16, Half>(source).ConvertToSingle(destination);

        /// <summary>
        /// Extracts metadata and input/output shapes from the ONNX model.
        /// </summary>
        private void GetOnnxMetaData()
            => OnnxData = _session.ToOnnxModel();

        public void Dispose()
        {
            _cudaGraph?.Dispose();
            _cudaGraph = null;
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

            _ortIoBinding?.Dispose();
            _session?.Dispose();
            _runOptions?.Dispose();

            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
