// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Core
{
    /// <summary>
    /// Initializes a new instance of the Yolo core class.
    /// </summary>
    internal class YoloCore(YoloOptions yoloOptions) : IDisposable
    {
        #region Fields
        private bool _isDisposed;

        public ArrayPool<float> customSizeFloatPool = default!;
        public ArrayPool<ushort> customSizeHalfPool = default!;
        public ArrayPool<ObjectResult> customSizeObjectResultPool = default!;
        private PinnedMemoryBufferPool _pinnedMemoryPool = default!;

        private readonly object _progressLock = new();

        private SKImageInfo _imageInfo;
        private long[] _inputShape = default!;
        private int _inputShapeSize;
        #endregion

        public OnnxModel OnnxModel { get; private set; } = default!;
        public YoloOptions YoloOptions { get => yoloOptions; init => yoloOptions = value; }
        public ModelType ModelType => OnnxModel.ModelType;

        /// <summary>
        /// Initializes the YOLO model with the specified model type.
        /// </summary>
        public void InitializeYolo()
        {
            if (YoloOptions.ExecutionProvider is null)
                throw new YoloDotNetModelException("Execution Provider is missing. Please add an execution provider.", nameof(YoloOptions));

            OnnxModel = yoloOptions.ExecutionProvider.OnnxData.GetOnnxProperties();

            VerifyExpectedModelType(OnnxModel.ModelType);

            _inputShape = OnnxModel.InputShape;
            _inputShapeSize = OnnxModel.InputShapeSize;

            // Create custom array pools based on model input size and output channels
            customSizeFloatPool = ArrayPool<float>.Create(CalculateBufferPoolSize(_inputShapeSize), 20); 
            customSizeHalfPool = ArrayPool<ushort>.Create(CalculateBufferPoolSize(_inputShapeSize), 20);

            // ObjectResult pool based on output channels
            customSizeObjectResultPool = ArrayPool<ObjectResult>.Create(CalculateBufferPoolSize(OnnxModel.Outputs[0].Channels), maxArraysPerBucket: 10);

            var format = (OnnxModel.Input.Channels == 1) ? SKColorType.Gray8 : SKColorType.Rgb888x;
            _imageInfo = new SKImageInfo(OnnxModel.Input.Width, OnnxModel.Input.Height, format, SKAlphaType.Opaque);

            _pinnedMemoryPool = new PinnedMemoryBufferPool(_imageInfo);

            FrameSaveService.Start();
        }

        /// <summary>
        /// Runs the YOLO model on the provided image and returns the inference results.
        /// </summary>
        /// <param name="image">The input image to process.</param>
        /// <returns>InferenceResult()</returns>
        public InferenceResult Run<T>(T image)
        {
            lock (_progressLock)
            {
            var pinnedBuffer = _pinnedMemoryPool.Rent();
                var normalizedPixelsFloatBuffer = customSizeFloatPool.Rent(_inputShapeSize);
                var normalizedPixelsUshortBuffer = customSizeHalfPool.Rent(_inputShapeSize);

            try
            {
                    // Resize image to model input size and store in pinned buffer for faster access
                    var originalImageSize = YoloOptions.ImageResize == ImageResize.Proportional
                        ? image.ResizeImageProportional(YoloOptions.SamplingOptions, pinnedBuffer)
                        : image.ResizeImageStretched(YoloOptions.SamplingOptions, pinnedBuffer);

                    // Declare struct to hold inference results
                    InferenceResult inferenceResult;

                    // Run inference using the selected execution provider and model data type
                    if (OnnxModel.ModelDataType == ModelDataType.Float16)
                    {
                        pinnedBuffer.Pointer.NormalizePixelsToArray(_inputShape, _inputShapeSize, normalizedPixelsUshortBuffer);
                        inferenceResult = YoloOptions.ExecutionProvider.Run<ushort>(normalizedPixelsUshortBuffer);
                    }
                    else
                    {
                        pinnedBuffer.Pointer.NormalizePixelsToArray(_inputShape, _inputShapeSize, normalizedPixelsFloatBuffer);
                        inferenceResult = YoloOptions.ExecutionProvider.Run<float>(normalizedPixelsFloatBuffer);
                    }

                    // Attach original image size to the inference result for use to calculate bounding boxes to original image size.
                    inferenceResult.ImageOriginalSize = originalImageSize;

                    return inferenceResult;
                }
            finally
            {
                    // Return rented buffers to their respective pools wuithout clearing for performance.
                    // WARNING: Only disable clearing if data is overwritten on next use to avoid data leakage!
                    customSizeFloatPool.Return(normalizedPixelsFloatBuffer, false);
                    customSizeHalfPool.Return(normalizedPixelsUshortBuffer, false);
                _pinnedMemoryPool.Return(pinnedBuffer);
            }
        }
        }

        #region Helper methods

        /// <summary>
        /// Removes overlapping bounding boxes in a list of object detection results.
        /// </summary>
        /// <param name="predictionSpan">A span with predition results</param>
        /// <param name="iouThreshold">Higher Iou-threshold result in fewer detections by excluding overlapping boxes.</param>
        /// <returns>A filtered list with non-overlapping bounding boxes based on confidence scores.</returns>
        public ObjectResult[] RemoveOverlappingBoxes(Span<ObjectResult> predictionSpan, double iouThreshold)
        {
            var totalPredictions = predictionSpan.Length;

            if (totalPredictions == 0)
                return [];

            // Sort by confidence
            MemoryExtensions.Sort(predictionSpan, ConfidenceComparer.Instance);

            var buffer = customSizeObjectResultPool.Rent(totalPredictions);

            try
            {
                var counter = 0;
                for (int i = 0; i < totalPredictions; i++)
                {
                    var item = predictionSpan[i];

                    bool overlapFound = false;
                    for (int j = 0; j < counter; j++)
                    {
                        if (CalculateIoU(item.BoundingBox, buffer[j].BoundingBox) > iouThreshold)
                        {
                            overlapFound = true;
                            break;
                        }
                    }

                    if (!overlapFound)
                        buffer[counter++] = item;
                }

                return [.. buffer.AsSpan(0, counter)];
            }
            finally
            {
                customSizeObjectResultPool.Return(buffer, false);
            }
        }

        /// <summary>
        /// Calculate buffer pool size as the next power of two to ensure array pool efficiency.
        /// </summary>
        public static int CalculateBufferPoolSize(int bufferSize) => 1 << (int)Math.Ceiling(Math.Log2(bufferSize));

        /// <summary>
        /// Squash value to a number between 0 and 1
        /// </summary>
        public static float Sigmoid(float value) => 1 / (1 + MathF.Exp(-value));

        /// <summary>
        /// Calculate pixel luminance
        /// </summary>
        public static byte CalculatePixelLuminance(float value) => (byte)(255 - value * 255);

        /// <summary>
        /// Calculate pixel by byte to confidence
        /// </summary>
        public static float CalculatePixelConfidence(byte value) => value / 255F;

        /// <summary>
        /// Calculate radian to degree
        /// </summary>
        public static float CalculateRadianToDegree(float value) => value * (180 / (float)Math.PI);

        /// <summary>
        /// Calculates the Intersection over Union (IoU) between two rectangles.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CalculateIoU(in SKRectI a, in SKRectI b)
        {
            // Use "in" keyword on parameters to pass by reference without allowing modification for better performance.

            int left = Math.Max(a.Left, b.Left);
            int top = Math.Max(a.Top, b.Top);
            int right = Math.Min(a.Right, b.Right);
            int bottom = Math.Min(a.Bottom, b.Bottom);

            int width = right - left;
            int height = bottom - top;

            if (width <= 0 || height <= 0)
                return 0f;

            int intersection = width * height;

            int areaA = (a.Right - a.Left) * (a.Bottom - a.Top);
            int areaB = (b.Right - b.Left) * (b.Bottom - b.Top);

            return (float)intersection / (areaA + areaB - intersection);
        }

        public (float, float, float, float) CalculateGain(SKSizeI size)
            => YoloOptions.ImageResize == ImageResize.Proportional ? CalculateProportionalGain(size) : CalculateStretchedGain(size);

        /// <summary>
        /// Calculates the padding and scaling factor needed to adjust the bounding box
        /// so that the detected object can be resized to match the original image size.
        /// </summary>
        /// <param name="size">The original image size.</param>
        public (float, float, float, float) CalculateProportionalGain(SKSizeI size)
        {
            var model = OnnxModel;

            var (w, h) = (size.Width, size.Height);

            var gain = Math.Max((float)w / model.Input.Width, (float)h / model.Input.Height);
            var ratio = Math.Min(model.Input.Width / (float)size.Width, model.Input.Height / (float)size.Height);
            var (xPad, yPad) = ((model.Input.Width - w * ratio) / 2, (model.Input.Height - h * ratio) / 2);

            return (xPad, yPad, gain, 0);
        }

        /// <summary>
        /// Calculates the padding and scaling factor needed to adjust the bounding box
        /// so that the detected object can be resized to match the original image size.
        /// </summary>
        /// <param name="size">The original image size.</param>
        public (float, float, float, float) CalculateStretchedGain(SKSizeI size)
        {
            var model = OnnxModel;

            var (w, h) = (size.Width, size.Height); // image w and h
            var (xGain, yGain) = (model.Input.Width / (float)w, model.Input.Height / (float)h); // x, y gains
            var (xPad, yPad) = ((model.Input.Width - w * xGain) / 2, (model.Input.Height - h * yGain) / 2); // left, right pads

            return (xPad, yPad, xGain, yGain);
        }

        /// <summary>
        /// Verify that loaded model is of the expected type
        /// </summary>
        public void VerifyExpectedModelType(ModelType expectedModelType)
        {
            if (expectedModelType.Equals(OnnxModel.ModelType) is false)
                throw new YoloDotNetModelMismatchException($"Loaded ONNX-model is of type {OnnxModel.ModelType} and can't be used for {expectedModelType}.");
        }

        /// <summary>
        /// Releases resources and suppresses the finalizer for the current object.
        /// </summary>
        public void Dispose()
        {
            if (_isDisposed)
                return;

            _pinnedMemoryPool?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
