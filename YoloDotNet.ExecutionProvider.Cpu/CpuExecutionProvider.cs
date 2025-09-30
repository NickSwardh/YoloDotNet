// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.Cpu
{
    public class CpuExecutionProvider : IExecutionProvider, IDisposable
    {
        public OnnxDataRecord OnnxData { get; private set; } = default!;

        #region Fields
        private static InferenceSession _session = default!;
        private RunOptions _runOptions = default!;
        private long[] _inputShape = default!;

        private float[] _outputBuffer0 = default!;
        private float[] _outputBuffer1 = default!;

        private TensorElementType _elementDataType = default!;
        private int _inputShapeSize;
        #endregion

        #region Constructors
        /// <summary>
        /// Constructs a CpuExecutionProvider for running ONNX models on the CPU.
        /// </summary>
        /// <param name="model"></param>
        public CpuExecutionProvider(string model)
        {
            InitializeYolo(model);
        }

        /// <summary>
        /// Constructs a CpuExecutionProvider for running ONNX models on the CPU.
        /// </summary>
        /// <param name="model"></param>
        public CpuExecutionProvider(byte[] model)
        {
            InitializeYolo(model);
        }
        #endregion

        #region Initialization
        private void InitializeYolo(object model)
        {
            ConfigureOrtEnv();

            _runOptions = new RunOptions();

            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableCpuMemArena = true
            };

            // Create session using bytes if available; else load from file with selected provider.
            _session = (model is byte[] modelBytes)
                ? new InferenceSession(modelBytes, options)
                : new InferenceSession((string)model, options);

            var metaData = _session.ModelMetadata.CustomMetadataMap;

            GetOnnxMetaData();
            AllocateOutputBuffers();

            // Set the input shape for creating tensors during inference.
            _inputShape = [.. OnnxData.InputShape.Select(i => (long)i)];
        }
        #endregion

        #region Run Inference
        /// <summary>
        /// Run inference on the provided normalized pixel data.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="normalizedPixels"></param>
        /// <returns></returns>
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

                // Run inference
                using var result = _session.Run(
                    _runOptions,
                    [OnnxData.InputName],
                    [inputOrtValue],
                    OnnxData.OutputNames);

                if (elementType == TensorElementType.Float)
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
            // Pre-allocate output buffers if the model uses Float16 data type.
            if (OnnxData.ModelDataType == ModelDataType.Float16)
                return;

            // Calculate the total number of elements for each output tensor and allocate buffers.
            var (items, elements) = (OnnxData.OutputShapes[0][1], OnnxData.OutputShapes[0][2]);
            _outputBuffer0 = new float[elements * items];

            // If there is a second output tensor, allocate a buffer for it as well.
            if (OnnxData.OutputShapes.Length == 2)
            {
                (items, elements) = (OnnxData.OutputShapes[1][1], OnnxData.OutputShapes[1][2]);
                _outputBuffer1 = new float[elements * items];
            }
        }

        private void ConvertFloat16ToFloat(ReadOnlySpan<Float16> tensorData0, ReadOnlySpan<Float16> tensorData1)
        {
            // Convert Float16 to float and store in pre-allocated buffers
            for (int i = 0; i < tensorData0.Length; i++)
                _outputBuffer0[i] = (float)tensorData0[i];

            for (int i = 0; i < tensorData1.Length; i++)
                _outputBuffer1[i] = (float)tensorData1[i];
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
            var inputSize = (int)ShapeUtils.GetSizeForShape(inputShape);

            _elementDataType = GetModelElementType();
            _inputShapeSize = (int)ShapeUtils.GetSizeForShape(inputShape);

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
        internal static TensorElementType GetModelElementType()
            => _session.InputMetadata["images"].ElementDataType;

        public void Dispose()
        {
            _session?.Dispose();
            _runOptions?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}