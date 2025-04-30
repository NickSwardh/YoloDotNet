namespace YoloDotNet.Core
{
    /// <summary>
    /// Initializes a new instance of the Yolo core class.
    /// </summary>
    /// <param name="onnxModel">The path to the ONNX model file to use.</param>
    /// <param name="useCuda">Indicates whether to use CUDA for GPU acceleration.</param>
    /// <param name="allocateGpuMemory">Indicates whether to allocate GPU memory.</param>
    /// <param name="gpuId">The GPU device ID to use when CUDA is enabled.</param>
    internal class YoloCore(string onnxModel, bool useCuda, bool allocateGpuMemory, int gpuId) : IDisposable
    {
        #region Fields
        private bool _isDisposed;

        private InferenceSession _session = default!;
        private RunOptions _runOptions = default!;
        private OrtIoBinding _ortIoBinding = default!;
        private int _tensorBufferSize;
        public ArrayPool<float> customSizeFloatPool = default!;
        public ArrayPool<ObjectResult> customSizeObjectResultPool = default!;

        private readonly object _progressLock = new();

        private SKImageInfo _imageInfo;

        public ParallelOptions parallelOptions = default!;
        #endregion

        public YoloOptions YoloOptions { get; private set; } = default!;
        public OnnxModel OnnxModel { get; private set; } = default!;

        /// <summary>
        /// Initializes the YOLO model with the specified model type.
        /// </summary>
        /// <param name="modelType">The type of the model to be initialized.</param>
        public void InitializeYolo(YoloOptions yoloOptions)
        {
            YoloOptions = yoloOptions;

            _session = useCuda
                ? new InferenceSession(onnxModel, SessionOptions.MakeSessionOptionWithCudaProvider(gpuId))
                : new InferenceSession(onnxModel);

            _runOptions = new RunOptions();
            _ortIoBinding = _session.CreateIoBinding();

            OnnxModel = _session.GetOnnxProperties();

            VerifyExpectedModelType(yoloOptions.ModelType);

            parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

            // tensorBufferSize can be calculated once and reused for all calls, as it is based on the model properties
            _tensorBufferSize = OnnxModel.Input.BatchSize * OnnxModel.Input.Channels * OnnxModel.Input.Width * OnnxModel.Input.Height;
            customSizeFloatPool = ArrayPool<float>.Create(maxArrayLength: _tensorBufferSize + 1, maxArraysPerBucket: 10);
            customSizeObjectResultPool = ArrayPool<ObjectResult>.Create(maxArrayLength: OnnxModel.Outputs[0].Channels + 1, maxArraysPerBucket: 10);

            _imageInfo = new SKImageInfo(OnnxModel.Input.Width, OnnxModel.Input.Height, SKColorType.Rgba8888, SKAlphaType.Opaque);

            if (useCuda && allocateGpuMemory)
                _session.AllocateGpuMemory(_ortIoBinding,
                    _runOptions,
                    customSizeFloatPool,
                    _imageInfo,
                    YoloOptions.SamplingOptions);

            // Run frame-save service
            FrameSaveService.Start();
        }

        /// <summary>
        /// Runs the YOLO model on the provided image and returns the inference results.
        /// </summary>
        /// <param name="image">The input image to process.</param>
        /// <returns>A read-only collection of OrtValue representing the inference results.</returns>
        public IDisposableReadOnlyCollection<OrtValue> Run(SKBitmap image)
        {
            //_resizedBitmap.Erase(SKColors.Black);

            using var resizedImage = YoloOptions.ImageResize == ImageResize.Proportional
                ? image.ResizeImageProportional(_imageInfo, YoloOptions.SamplingOptions)
                : image.ResizeImageStretched(_imageInfo, YoloOptions.SamplingOptions);

            var tensorArrayBuffer = customSizeFloatPool.Rent(minimumLength: _tensorBufferSize);

            try
            {
                lock (_progressLock)
                {
                    var tensorPixels = resizedImage.NormalizePixelsToTensor(OnnxModel.InputShape, _tensorBufferSize, tensorArrayBuffer);

                    using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensorPixels.Buffer, OnnxModel.InputShape);

                    var inputNames = new Dictionary<string, OrtValue>
                    {
                        { OnnxModel.InputName, inputOrtValue }
                    };

                    return _session.Run(_runOptions, inputNames, OnnxModel.OutputNames);
                }
            }
            finally
            {
                customSizeFloatPool.Return(tensorArrayBuffer, true);
            }
        }

        /// <summary>
        /// Removes overlapping bounding boxes in a list of object detection results.
        /// </summary>
        /// <param name="predictions">The list of object detection results to process.</param>
        /// <param name="iouThreshold">Higher Iou-threshold result in fewer detections by excluding overlapping boxes.</param>
        /// <returns>A filtered list with non-overlapping bounding boxes based on confidence scores.</returns>
        public ObjectResult[] RemoveOverlappingBoxes(ObjectResult[] predictions, double iouThreshold)
        {
            Array.Sort(predictions, (a, b) => b.Confidence.CompareTo(a.Confidence));
            var result = new HashSet<ObjectResult>();

            var predictionSpan = predictions.AsSpan();
            var totalPredictions = predictionSpan.Length;

            for (int i = 0; i < totalPredictions; i++)
            {
                var item = predictionSpan[i];

                bool overlapFound = false;
                foreach (var res in result)
                {
                    if (CalculateIoU(item.BoundingBox, res.BoundingBox) > iouThreshold)
                    {
                        overlapFound = true;
                        break;
                    }
                }
                if (!overlapFound)
                {
                    result.Add(item);
                }
            }

            return [.. result];
        }

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
        public static float CalculatePixelConfidence(byte value) => 1 - value / 255F;

        /// <summary>
        /// Calculate radian to degree
        /// </summary>
        /// <param name="value"></param>
        public static float CalculateRadianToDegree(float value) => value * (180 / (float)Math.PI);

        /// <summary>
        /// Calculate rectangle area
        /// </summary>
        /// <param name="rect"></param>
        public static float CalculateArea(SKRectI rect) => rect.Width * rect.Height;

        /// <summary>
        /// Calculate IoU (Intersection Over Union) bounding box overlap.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        public static float CalculateIoU(SKRectI a, SKRectI b)
        {
            if (a.IntersectsWith(b) is false) // Quick check before calculating intersection
                return 0;

            var intersectionArea = CalculateArea(SKRectI.Intersect(a, b));

            return intersectionArea == 0
                ? 0
                : intersectionArea / (CalculateArea(a) + CalculateArea(b) - intersectionArea);
        }

        public (float, float, float, float) CalculateGain(SKBitmap image)
            => YoloOptions.ImageResize == ImageResize.Proportional ? CalculateProportionalGain(image) : CalculateStretchedGain(image);

        /// <summary>
        /// Calculates the padding and scaling factor needed to adjust the bounding box
        /// so that the detected object can be resized to match the original image size.
        /// </summary>
        /// <param name="image">The image for which the bounding box needs to be adjusted.</param>
        public (float, float, float, float) CalculateProportionalGain(SKBitmap image)
        {
            var model = OnnxModel;

            var (w, h) = (image.Width, image.Height);

            var gain = Math.Max((float)w / model.Input.Width, (float)h / model.Input.Height);
            var ratio = Math.Min(model.Input.Width / (float)image.Width, model.Input.Height / (float)image.Height);
            var (xPad, yPad) = ((model.Input.Width - w * ratio) / 2, (model.Input.Height - h * ratio) / 2);

            return (xPad, yPad, gain, 0);
        }

        /// <summary>
        /// Calculates the padding and scaling factor needed to adjust the bounding box
        /// so that the detected object can be resized to match the original image size.
        /// </summary>
        /// <param name="image">The image for which the bounding box needs to be adjusted.</param>
        public (float, float, float, float) CalculateStretchedGain(SKBitmap image)
        {
            var model = OnnxModel;

            var (w, h) = (image.Width, image.Height); // image w and h
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
                throw new Exception($"Loaded ONNX-model is of type {OnnxModel.ModelType} and can't be used for {expectedModelType}.");
        }

        /// <summary>
        /// Releases resources and suppresses the finalizer for the current object.
        /// </summary>
        public void Dispose()
        {
            if (_isDisposed)
                return;

            _session?.Dispose();
            _ortIoBinding?.Dispose();
            _runOptions?.Dispose();

            ImageExtension.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}
