// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models.Interfaces
{
    /// <summary>
    /// Interface for execution providers to implement for running inference on ONNX models.
    /// </summary>
    public interface IExecutionProvider
    {
        /// <summary>
        /// Method to run inference on the model with the provided normalized pixel data.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="normalizedPixels"></param>
        /// <returns></returns>
        public InferenceResult Run<T>(T[] normalizedPixels) where T : unmanaged;

        /// <summary>
        /// Record containing metadata about the ONNX model.
        /// </summary>
        public OnnxDataRecord OnnxData { get; }
    }
}
