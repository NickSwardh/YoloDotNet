// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.Cpu
{
    public class CpuExecutionProvider : ICpu, IExecutionProvider, IDisposable
    {
        public OnnxDataRecord OnnxData { get; private set; } = default!;

        private InferenceSession _session = default!;
        private RunOptions _runOptions = default!;

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

            OnnxData = new OnnxDataRecord(
                metaData,
                _session.InputNames[0],
                [.. _session.OutputNames],
                _session.InputMetadata.Values.Select(x => x.Dimensions).FirstOrDefault() ?? [],
                [.. _session.OutputMetadata.Values.Select(x => x.Dimensions)],
                metaData["names"]
            );
        }

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
                [.. OnnxData.InputShape.Select(i => (long)i)]
                );

            // Run the inference session with the input OrtValue and retrieve the results.
            using var result = _session.Run(
                _runOptions,
                [OnnxData.InputName],
                [inputOrtValue],
                OnnxData.OutputNames);

            // Extract the tensor data from the results and return as an InferenceResult.
            var tensorData0 = result[0].GetTensorDataAsSpan<float>();
            var tensorData1 = ReadOnlySpan<float>.Empty;

            if (result.Count == 2)
                tensorData1 = result[1].GetTensorDataAsSpan<float>();

            return new InferenceResult(tensorData0, tensorData1);
        }

        public void Dispose()
        {
            _session?.Dispose();
            _runOptions?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}