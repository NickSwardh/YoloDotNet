// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.MIGraphX
{
    public class MIGraphXExecutionProvider : IExecutionProvider, IDisposable
    {
        public OnnxModel OnnxData { get; private set; } = default!;
        public object Session => _session;

        #region Private Fields
        private InferenceSession _session = default!;
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
        #endregion

        #region Constructors
        /// <summary>
        /// Constructs a MIGraphXExecutionProvider for running ONNX models using AMD MIGraphX.
        /// </summary>
        /// <param name="model">Path to the ONNX model file</param>
        /// <param name="deviceId">GPU device ID (default: 0)</param>
        public MIGraphXExecutionProvider(string model, int deviceId = 0)
        {
            InitializeYolo(model, deviceId);
        }

        /// <summary>
        /// Overload: Constructs a MIGraphXExecutionProvider for running ONNX models using AMD MIGraphX.
        /// </summary>
        /// <param name="model">ONNX model as byte array</param>
        /// <param name="deviceId">GPU device ID (default: 0)</param>
        public MIGraphXExecutionProvider(byte[] model, int deviceId = 0)
        {
            InitializeYolo(model, deviceId);
        }
        #endregion

        #region Initialization
        /// <summary>
        /// Initializes the ONNX Runtime session, configures the MIGraphX execution provider and allocates resources.
        /// </summary>
        /// <param name="model">ONNX model (file path or byte array)</param>
        /// <param name="deviceId">GPU device ID</param>
        private void InitializeYolo(object model, int deviceId)
        {
            ConfigureOrtEnv();

            var options = CreateSessionOptions(deviceId);

            // Create session using bytes if available; else load from file with selected provider.
            _session = (model is byte[] modelBytes)
                ? new InferenceSession(modelBytes, options)
                : new InferenceSession((string)model, options);

            GetOnnxMetaData();
            AllocateOutputBuffers();
            InitializeInferenceParameters();

            _runOptions = new RunOptions();
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
        /// <param name="normalizedPixels">Normalized pixel data</param>
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

        #region MIGraphX helper methods
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
        private SessionOptions CreateSessionOptions(int deviceId)
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            if (deviceId >= 0)
            {
                ConfigureMIGraphX(deviceId, options);
            }
            else if (deviceId == -1)
            {
                options.EnableCpuMemArena = true;
            }
            else
            {
                throw new ArgumentOutOfRangeException(
                    paramName: nameof(deviceId),
                    actualValue: deviceId,
                    message: "The specified deviceId is not valid. Use -1 for CPU execution, or 0 and above for a GPU device ID.");
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
        /// Configures the session options to use the MIGraphX execution provider with specified options.
        /// </summary>
        private static void ConfigureMIGraphX(int deviceId, SessionOptions options)
        {
            var migraphxOptions = new Dictionary<string, string>
            {
                { "device_id", deviceId.ToString() },
                // Specifies which GPU device to use (default = 0 if not set).

                { "migraphx_fp16_enable", "1" },
                // Enable FP16 precision for better performance on supported hardware

                { "migraphx_int8_enable", "0" },
                // Disable INT8 by default (can be enabled if model supports quantization)
            };

            options.AppendExecutionProvider("MIGraphX", migraphxOptions);
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
            _session?.Dispose();
            _runOptions?.Dispose();
            _currentResult?.Dispose();

            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
