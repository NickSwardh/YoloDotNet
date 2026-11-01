// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider
{
    /// <summary>
    /// Reads the model information from an ONNX Runtime session through its typed API, so it works in trimmed and AOT apps.
    /// Shared by all execution providers: each one compiles this file against its own ONNX Runtime package
    /// (through the YoloDotNet.ExecutionProvider.Shared shared project), so the YoloDotNet core needs no ONNX Runtime reference.
    /// </summary>
    internal static class InferenceSessionExtensions
    {
        public static OnnxModel ToOnnxModel(this InferenceSession session)
        {
            var inputs = session.InputMetadata.ToDictionary(x => x.Key, x => x.Value.Dimensions.Select(d => (long)d).ToArray());
            var outputs = session.OutputMetadata.ToDictionary(x => x.Key, x => x.Value.Dimensions.ToArray());
            var metadata = new Dictionary<string, string>(session.ModelMetadata.CustomMetadataMap);

            // The first input decides the data type
            var modelDataType = session.InputMetadata.Values.FirstOrDefault()?.ElementDataType == TensorElementType.Float16
                ? ModelDataType.Float16
                : ModelDataType.Float;

            return ParseOnnxData.CreateOnnxModel(inputs, outputs, metadata, modelDataType);
        }
    }
}
