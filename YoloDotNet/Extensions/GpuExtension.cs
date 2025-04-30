﻿namespace YoloDotNet.Extensions
{
    public static class GpuExtension
    {
        /// <summary>
        /// Allocate GPU memory for input data and ensure memory synchronization.
        /// </summary>
        public static void AllocateGpuMemory(this InferenceSession session,
            OrtIoBinding ortIoBinding,
            RunOptions runOptions,
            ArrayPool<float> customSizeFloatPool,
            SKImageInfo resizeInfo,
            SKSamplingOptions samplingOptions)
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

            // Run inference on the OrtIoBinding and bind allocated GPU-memory.
            session.RunWithBinding(runOptions, ortIoBinding);

            // Ensure that the output data is properly synchronized with memory after running the inference.
            ortIoBinding.SynchronizeBoundOutputs();

            // Initialize.
            session.InitializeGpu(customSizeFloatPool, resizeInfo, samplingOptions);
        }

        
        private static void InitializeGpu(this InferenceSession session, ArrayPool<float> customSizeFloatPool, SKImageInfo resizeInfo, SKSamplingOptions samplingOptions)
        {
            // Get model data from session
            var inputName = session.InputNames[0];
            var outputNames = session.OutputNames;
            var dimensions = session.InputMetadata[session.InputNames[0]].Dimensions;
            var (batchSize, channels, width, height) = (dimensions[0], dimensions[1], dimensions[2], dimensions[3]);

            // Create blank image for initial inference
            //using var img = SKImage.Create(new SKImageInfo(ImageConfig.GPU_IMG_ALLOC_SIZE, ImageConfig.GPU_IMG_ALLOC_SIZE));
            using var img = new SKBitmap(new SKImageInfo(ImageConfig.GPU_IMG_ALLOC_SIZE, ImageConfig.GPU_IMG_ALLOC_SIZE));

            using var resizedImg = img.ResizeImageProportional(resizeInfo, samplingOptions);

            // Prepare tensor buffer
            var tensorBufferSize = batchSize * channels * width * height;
            var tensorArrayBuffer = customSizeFloatPool.Rent(tensorBufferSize);

            var normalizedTensorPixels = resizedImg.NormalizePixelsToTensor([batchSize, channels, width, height], tensorBufferSize, tensorArrayBuffer);

            var inputShape = new long[]
            {
                batchSize,  // Batches (nr of images the model can process)
                channels,   // Total color channels the model expects
                width,      // Required image width
                height,     // Required image Height
            };

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, normalizedTensorPixels.Buffer, inputShape);

            var inputNames = new Dictionary<string, OrtValue>
            {
                { inputName, inputOrtValue }
            };

            using var ortResults = session.Run(new RunOptions(), inputNames, outputNames);

            // Return array buffer to array pool for reuse.
            customSizeFloatPool.Return(tensorArrayBuffer, true);
        }
    }
}
