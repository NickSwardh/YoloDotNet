// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using YoloDotNet.Exceptions;
using YoloDotNet.Extensions;

namespace YoloDotNet.ExecutionProvider.OpenVino
{
    public class OpenVinoExecutionProvider : IExecutionProvider, IDisposable
    {
        public OnnxModel OnnxData { get; private set; } = default!;
        public Object Session => _session;

        #region Fields
        private InferenceSession _session = default!;
        private RunOptions _runOptions = default!;

        private float[] _outputBuffer0 = default!;
        private float[] _outputBuffer1 = default!;

        private string[] _inputNames = default!;
        private long[] _inputShape = default!;
        private int _inputShapeSize;
        private List<string> _outputNames = default!;
        private TensorElementType _elementDataType = default!;
        private int _dataTypeSize;

        private IDisposableReadOnlyCollection<OrtValue>? _currentResult;
        #endregion

        #region Constructors
        /// <summary>
        /// Constructs a OpenVinoExecutionProvider for running ONNX models using Intel GPUs.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="openVino"></param>
        public OpenVinoExecutionProvider(string model, OpenVino? openVino = null)
        {
            InitializeYolo(model, openVino);
        }

        /// <summary>
        /// Constructs a OpenVinoExecutionProvider for running ONNX models using Intel GPUs.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="openVino"></param>
        public OpenVinoExecutionProvider(object model, OpenVino? openVino = null)
        {
            InitializeYolo(model, openVino);
        }
        #endregion

        #region Initialization
        private void InitializeYolo(object model, OpenVino? openVino)
        {
            ConfigureOrtEnv();

            var options = CreateSessionOptions(openVino);

            _session = (model is byte[] modelBytes)
                ? new InferenceSession(modelBytes, options)
                : new InferenceSession((string)model, options);

            _runOptions = new RunOptions();

            GetOnnxMetaData();
            AllocateOutputBuffers();
            InitializeInferenceParameters();

            //_inputShape = [.. OnnxData.InputShape.Select(i => (long)i)];
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

        #region OpenVino helper methods
        private static SessionOptions CreateSessionOptions(OpenVino? openVino)
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            if (openVino is null
                || string.IsNullOrEmpty(openVino.DeviceType)
                || openVino.DeviceType.StartsWith("CPU", StringComparison.OrdinalIgnoreCase))
            {
                options.EnableCpuMemArena = true;
            }
            else
            {
                var ovOptions = new Dictionary<string, string>
                {
                    // OpenVINO EP supports device_type, precision, num_threads, num_streams, model_priority
                    ["device_type"] = openVino.DeviceType.ToUpper(),

                    // Precision mode: FP16 for speed, FP32 for accuracy. ACCURACY to use model-defined precision.
                    ["precision"] = openVino.Precision.ToString().ToUpper(), 

                    // Threading and streams
                    ["num_of_threads"] = openVino.Threads.ToString(), // Open Vino default to 8 threads.
                    ["num_streams"] = openVino.Streams.ToString(), // Open Vino default to 1 stream.

                    // Cache directory for storing compiled models
                    ["cache_dir"] = openVino.CachePath.ToString(),

                    // Performance tuning options
                    ["disable_dynamic_shapes"] = "True",
                    ["model_priority"] = openVino.ModelPriority.ToString().ToUpper()
                };

                options.AppendExecutionProvider("OpenVINOExecutionProvider", ovOptions);
            }

            return options;
        }

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
            => OnnxData = _session.ParseOnnx();

        /// <summary>
        /// Gets the tensor element type used by the model (e.g., Float32 or Float16).
        /// </summary>
        internal TensorElementType GetModelElementType()
            => _session.InputMetadata["images"].ElementDataType;

        private static void ConfigureOrtEnv()
        {
            try
            {
                var envOptions = new EnvironmentCreationOptions
                {
                    logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
                };

                OrtEnv.CreateInstanceWithOptions(ref envOptions);
            }
            catch (OnnxRuntimeException ex) when (ex.Message.Contains("OrtEnv singleton instance already exists"))
            {
                // OrtEnv has already been initialized - ignore and continue gracefully...
            }
        }

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
