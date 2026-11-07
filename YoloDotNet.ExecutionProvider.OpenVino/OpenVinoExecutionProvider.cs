// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.OpenVino
{
    public class OpenVinoExecutionProvider : IExecutionProvider, IDisposable
    {
        public OnnxModel OnnxData { get; private set; } = default!;
        public Object Session => _session;

        #region Fields
        private InferenceSession _session = default!;
        private RunOptions _runOptions = default!;
        private readonly object _runLock = new();

        private string[] _inputNames = default!;
        private long[] _inputShape = default!;
        private int _inputShapeSize;
        private List<string> _outputNames = default!;
        private TensorElementType _elementDataType = default!;
        private int _dataTypeSize;
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

                // Run inference. The output belongs to this call only: it travels with the returned InferenceResult
                // and is released when the caller disposes it, so concurrent calls never share or free each other's output.
                IDisposableReadOnlyCollection<OrtValue> outputs;

                // Concurrent Run calls on one session have not been verified for the OpenVINO execution provider, so
                // inference on this session is serialized. Only the inference call is locked: resizing, normalization
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

                    // Precision mode: FP16 for speed, FP32 for accuracy.
                    ["precision"] = openVino.Precision.ToString().ToUpper(),

                    // Threading and streams
                    ["num_of_threads"] = openVino.Threads.ToString(), // Open Vino default to 8 threads.
                    ["num_streams"] = openVino.Streams.ToString(), // Open Vino default to 1 stream.

                    // Performance tuning options
                    ["disable_dynamic_shapes"] = "True",
                    ["model_priority"] = openVino.ModelPriority.ToString().ToUpper()
                };

                // Cache directory for compiled models. Optional: without it, OpenVINO compiles the model on every start.
                if (string.IsNullOrWhiteSpace(openVino.CachePath) is false)
                    ovOptions["cache_dir"] = openVino.CachePath;

                options.AppendExecutionProvider("OpenVINOExecutionProvider", ovOptions);
            }

            return options;
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

            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
