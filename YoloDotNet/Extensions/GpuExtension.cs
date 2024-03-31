namespace YoloDotNet.Extensions
{
    public static class GpuExtension
    {
        /// <summary>
        /// Allocates GPU memory for input data and ensures memory synchronization.
        /// </summary>
        public static void AllocateGpuMemory(this InferenceSession session, OrtIoBinding ortIoBinding, RunOptions runOptions)
        {
            // Get input shape.
            var inputShape = Array.ConvertAll(session.InputMetadata[session.InputNames[0]].Dimensions, Convert.ToInt64);

            // Calculate input size.
            var inputSizeInBytes = ShapeUtils.GetSizeForShape(inputShape) * sizeof(float);

            // Allocates unmanaged memory.
            IntPtr allocPtr = Marshal.AllocHGlobal((int)inputSizeInBytes);

            // Create OrtValue with the allocated memory as the data buffer.
            using (var ortValueTensor = OrtValue.CreateTensorValueWithData(
                OrtMemoryInfo.DefaultInstance,
                TensorElementType.Float,
                inputShape, allocPtr, inputSizeInBytes))
            {
                ortIoBinding.BindInput(session.InputNames[0], ortValueTensor);
            }
            
            // Bind output
            ortIoBinding.BindOutputToDevice(session.OutputNames[0], OrtMemoryInfo.DefaultInstance);

            // Ensure input data is properly synchronized with memory before running the inference.
            ortIoBinding.SynchronizeBoundInputs();

            // Run inference and bind allocated GPU-memory.
            session.RunWithBinding(runOptions, ortIoBinding);

            // Ensure that the output data is properly synchronized with memory after running the inference.
            ortIoBinding.SynchronizeBoundOutputs();
        }
    }
}
