// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.Cuda
{
    public static class GpuExtension
    {
        /// <summary>
        /// Allocate GPU memory for input data and ensure memory synchronization.
        /// </summary>
        public static void AllocateGpuMemory(this InferenceSession session,
            OrtIoBinding ortIoBinding,
            RunOptions runOptions,
            TensorElementType tensorElementType)
        {
            // Get input shape.
            var inputShape = Array.ConvertAll(session.InputMetadata[session.InputNames[0]].Dimensions, Convert.ToInt64);

            // Determine byte size based on model data type.
            var byteSize = tensorElementType == TensorElementType.Float ? sizeof(float) : sizeof(ushort);

            // Calculate input size.
            var inputSizeInBytes = ShapeUtils.GetSizeForShape(inputShape) * byteSize;

            // Allocates unmanaged memory.
            nint allocPtr = Marshal.AllocHGlobal((int)inputSizeInBytes);

            // Create OrtValue with the allocated memory as the data buffer.
            using (var ortValueTensor = OrtValue.CreateTensorValueWithData(
                OrtMemoryInfo.DefaultInstance,
                tensorElementType,
                inputShape,
                allocPtr,
                inputSizeInBytes))
            {
                ortIoBinding.BindInput(session.InputNames[0], ortValueTensor);
            }

            // Bind output
            ortIoBinding.BindOutputToDevice(session.OutputNames[0], OrtMemoryInfo.DefaultInstance);

            // Ensure input data is properly synchronized with memory before running the inference.
            ortIoBinding.SynchronizeBoundInputs();

            // Run inference on the OrtIoBinding and bind allocated GPU-memory.
            session.RunWithBinding(runOptions, ortIoBinding);

            // Ensure that the output data is properly synchronized with memory after running the inference.
            ortIoBinding.SynchronizeBoundOutputs();
        }
    }
}
