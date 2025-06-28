namespace YoloDotNet.Core
{
    /// <summary>
    /// Initializes a new instance of the Yolo core class.
    /// </summary>
    internal class YoloCore(YoloOptions yoloOptions) : IDisposable
    {
        #region Fields
        private bool _isDisposed;

        private InferenceSession _session = default!;
        private RunOptions _runOptions = default!;
        private OrtIoBinding _ortIoBinding = default!;
        private int _tensorBufferSize;
        public ArrayPool<float> customSizeFloatPool = default!;
        public ArrayPool<ObjectResult> customSizeObjectResultPool = default!;
        private PinnedMemoryBufferPool _pinnedMemoryPool = default!;

        private Dictionary<string, OrtValue> _inputNames = default!;
        private readonly object _progressLock = new();

        private SKImageInfo _imageInfo;
        public ParallelOptions parallelOptions = default!;
        #endregion

        public OnnxModel OnnxModel { get; private set; } = default!;
        public YoloOptions YoloOptions { get => yoloOptions; init => yoloOptions = value; }
        public ModelType ModelType => OnnxModel.ModelType;

        /// <summary>
        /// Initializes the YOLO model with the specified model type.
        /// </summary>
        public void InitializeYolo()
        {
            if (string.IsNullOrEmpty(YoloOptions.OnnxModel) && YoloOptions.OnnxModelBytes is null)
                throw new ArgumentException("No ONNX model was specified. Please provide a model path or byte array.", nameof(YoloOptions));

            InjectModelIntoOnnxRuntime();

            _runOptions = new RunOptions();
            _ortIoBinding = _session.CreateIoBinding();

            OnnxModel = _session.GetOnnxProperties();

            VerifyExpectedModelType(ModelType);

            parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

            // tensorBufferSize can be calculated once and reused for all calls, as it is based on the model properties
            _tensorBufferSize = OnnxModel.Input.BatchSize * OnnxModel.Input.Channels * OnnxModel.Input.Width * OnnxModel.Input.Height;
            customSizeFloatPool = ArrayPool<float>.Create(maxArrayLength: _tensorBufferSize + 1, maxArraysPerBucket: 10);
            customSizeObjectResultPool = ArrayPool<ObjectResult>.Create(maxArrayLength: OnnxModel.Outputs[0].Channels + 1, maxArraysPerBucket: 10);

            _imageInfo = new SKImageInfo(OnnxModel.Input.Width, OnnxModel.Input.Height, SKColorType.Rgb888x, SKAlphaType.Opaque);

            _pinnedMemoryPool = new PinnedMemoryBufferPool(_imageInfo);

            if (YoloOptions.Cuda && YoloOptions.PrimeGpu)
                _session.AllocateGpuMemory(_ortIoBinding,
                    _runOptions,
                    customSizeFloatPool,
                    _pinnedMemoryPool,
                    YoloOptions.SamplingOptions);

            _inputNames = new Dictionary<string, OrtValue>
            {
                { OnnxModel.InputName, null! }
            };

            // Run frame-save service
            FrameSaveService.Start();
        }

        private void InjectModelIntoOnnxRuntime()
        {
            // Create session options if using CUDA
            SessionOptions sessionOptions = new SessionOptions {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            };

            if (YoloOptions.Cuda)
            {
                //sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(YoloOptions.GpuId);
                sessionOptions.AppendExecutionProvider_CUDA(YoloOptions.GpuId);
            }

            // Create inference session from byte[] or file path
            _session = (YoloOptions.OnnxModelBytes != null && YoloOptions.OnnxModelBytes.Length > 0)
                ? new InferenceSession(YoloOptions.OnnxModelBytes, sessionOptions)
                : new InferenceSession(YoloOptions.OnnxModel, sessionOptions);
        }

        /// <summary>
        /// Runs the YOLO model on the provided image and returns the inference results.
        /// </summary>
        /// <param name="image">The input image to process.</param>
        /// <returns>A read-only collection of OrtValue representing the inference results.</returns>
        public (IDisposableReadOnlyCollection<OrtValue>, SKSizeI) Run<T>(T image)
        {
            var tensorArrayBuffer = customSizeFloatPool.Rent(minimumLength: _tensorBufferSize);
            var pinnedBuffer = _pinnedMemoryPool.Rent();

            try
            {
                lock (_progressLock)
                {

                    var (pointer, imageSize) = YoloOptions.ImageResize == ImageResize.Proportional
                        ? image.ResizeImageProportional(YoloOptions.SamplingOptions, pinnedBuffer)
                        : image.ResizeImageStretched(YoloOptions.SamplingOptions, pinnedBuffer);

                    var tensorPixels = pointer.NormalizePixelsToTensor(OnnxModel.InputShape, _tensorBufferSize, tensorArrayBuffer);
                    using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensorPixels.Buffer, OnnxModel.InputShape);

                    _inputNames[OnnxModel.InputName] = inputOrtValue;
                    return (_session.Run(_runOptions, _inputNames, OnnxModel.OutputNames), imageSize);
                }
            }
            finally
            {
                customSizeFloatPool.Return(tensorArrayBuffer, true);
                _pinnedMemoryPool.Return(pinnedBuffer);
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
        ///public static float CalculatePixelConfidence(byte value) => 1 - value / 255F;
        public static float CalculatePixelConfidence(byte value) => value / 255F;

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

        public (float, float, float, float) CalculateGain(SKSizeI size)
            => YoloOptions.ImageResize == ImageResize.Proportional ? CalculateProportionalGain(size) : CalculateStretchedGain(size);

        /// <summary>
        /// Calculates the padding and scaling factor needed to adjust the bounding box
        /// so that the detected object can be resized to match the original image size.
        /// </summary>
        /// <param name="image">The image for which the bounding box needs to be adjusted.</param>
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
        /// <param name="image">The image for which the bounding box needs to be adjusted.</param>
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
            _pinnedMemoryPool?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}
