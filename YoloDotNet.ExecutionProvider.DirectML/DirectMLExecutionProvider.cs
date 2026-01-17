// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.DirectML
{
    public class DirectMLExecutionProvider : IExecutionProvider, IDisposable
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
        /// Constructs a DirectMLExecutionProvider for running ONNX models using DirectML GPU acceleration.
        /// </summary>
        /// <param name="model">Path to the ONNX model file.</param>
        /// <param name="gpuId">GPU device ID (0 = default GPU, -1 = CPU fallback).</param>
        public DirectMLExecutionProvider(string model, int gpuId = 0)
        {
            InitializeYolo(model, gpuId);
        }

        /// <summary>
        /// Constructs a DirectMLExecutionProvider for running ONNX models using DirectML GPU acceleration.
        /// </summary>
        /// <param name="model">ONNX model as byte array.</param>
        /// <param name="gpuId">GPU device ID (0 = default GPU, -1 = CPU fallback).</param>
        public DirectMLExecutionProvider(byte[] model, int gpuId = 0)
        {
            InitializeYolo(model, gpuId);
        }
        #endregion

        #region Initialization
        private void InitializeYolo(object model, int gpuId)
        {
            ConfigureOrtEnv();

            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                
                // DirectML requires sequential execution mode (parallel execution is not supported)
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,

                EnableCpuMemArena = true
            };

            // Use DirectML execution provider if a valid GPU ID is provided; else use CPU.
            if (gpuId >= 0)
            {
                options.AppendExecutionProvider_DML(gpuId);
            }
            else
            {
                options.EnableCpuMemArena = true;
            }

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
        /// Run inference on the provided normalized pixel data.
        /// </summary>
        /// <typeparam name="T">Data type (float or Float16).</typeparam>
        /// <param name="normalizedPixels">Normalized pixel data.</param>
        /// <returns>Inference results.</returns>
        unsafe public InferenceResult Run<T>(T[] normalizedPixels) where T : unmanaged
        {
            // Pin the ushort[] so we can get a raw pointer
            fixed (T* pData = normalizedPixels)
            {
                var elementType = (typeof(T) == typeof(float)) ? TensorElementType.Float : TensorElementType.Float16;

                using var inputOrtValue = OrtValue.CreateTensorValueWithData(
                    OrtMemoryInfo.DefaultInstance,
                    elementType,   // 👈 force ONNX to interpret buffer as Float16
                    _inputShape,
                    (IntPtr)pData,
                    _inputShapeSize * sizeof(T) // size in bytes (ushort is 2 bytes, float is 4 bytes)
                );

                // Dispose previous result before running new inference to prevent memory leaks
                _currentResult?.Dispose();

                // Run inference
                _currentResult = _session.Run(
                    _runOptions,
                    _inputNames,
                    [inputOrtValue],
                    _outputNames);

                if (elementType == TensorElementType.Float)
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

        #region Helper methods

        /// <summary>
        /// Configures the ONNX Runtime environment to log errors and fatals only.
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
        /// Allocates output buffers for float16 models only.
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

        unsafe private void ConvertFloat16ToFloat(ReadOnlySpan<Float16> tensorData0, ReadOnlySpan<Float16> tensorData1)
        {
            // Convert Float16 to float and store in pre-allocated buffers
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

        /// <summary>
        /// Gets the tensor element type used by the model (e.g., Float32 or Float16).
        /// </summary>
        internal TensorElementType GetModelElementType()
            => _session.InputMetadata["images"].ElementDataType;

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