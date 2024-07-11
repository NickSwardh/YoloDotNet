namespace YoloDotNet.Data
{
    /// <summary>
    /// Initializes a new instance of the Yolo core class.
    /// </summary>
    /// <param name="onnxModel">The path to the ONNX model file to use.</param>
    /// <param name="useCuda">Indicates whether to use CUDA for GPU acceleration.</param>
    /// <param name="allocateGpuMemory">Indicates whether to allocate GPU memory.</param>
    /// <param name="gpuId">The GPU device ID to use when CUDA is enabled.</param>
    public class YoloCore(string onnxModel, bool useCuda, bool allocateGpuMemory, int gpuId) : IDisposable
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

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
        public OnnxModel OnnxModel { get; private set; } = default!;

        /// <summary>
        /// Initializes the YOLO model with the specified model type.
        /// </summary>
        /// <param name="modelType">The type of the model to be initialized.</param>
        public void InitializeYolo(ModelType modelType)
        {
            _session = useCuda
                ? new InferenceSession(onnxModel, SessionOptions.MakeSessionOptionWithCudaProvider(gpuId))
                : new InferenceSession(onnxModel);

            _runOptions = new RunOptions();
            _ortIoBinding = _session.CreateIoBinding();

            OnnxModel = _session.GetOnnxProperties();

            VerifyExpectedModelType(modelType);

            parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

            // tensorBufferSize can be calculated once and reused for all calls, as it is based on the model properties
            _tensorBufferSize = OnnxModel.Input.BatchSize * OnnxModel.Input.Channels * OnnxModel.Input.Width * OnnxModel.Input.Height;
            customSizeFloatPool = ArrayPool<float>.Create(maxArrayLength: _tensorBufferSize + 1, maxArraysPerBucket: 10);
            customSizeObjectResultPool = ArrayPool<ObjectResult>.Create(maxArrayLength: OnnxModel.Outputs[0].Channels + 1, maxArraysPerBucket: 10);

            _imageInfo = new SKImageInfo(OnnxModel.Input.Width, OnnxModel.Input.Height, SKColorType.Rgb888x, SKAlphaType.Opaque);

            if (useCuda && allocateGpuMemory)
                _session.AllocateGpuMemory(_ortIoBinding, _runOptions, customSizeFloatPool, _imageInfo);
        }

        /// <summary>
        /// Runs the YOLO model on the provided image and returns the inference results.
        /// </summary>
        /// <param name="image">The input image to process.</param>
        /// <returns>A read-only collection of OrtValue representing the inference results.</returns>
        public IDisposableReadOnlyCollection<OrtValue> Run(SKImage image)
        {
            using var resizedImage = image.ResizeImage(_imageInfo);

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
        /// Runs inference on video data using the specified options and optional thresholds.
        /// Triggers events for progress, completion, and status changes during video processing.
        /// </summary>
        /// <typeparam name="T">The type of the inference results.</typeparam>
        /// <param name="options">Options for configuring video processing.</param>
        /// <param name="confidence">Confidence threshold for inference.</param>
        /// <param name="iouThreshold">IoU threshold value for excluding bounding boxes.</param>
        /// <param name="func">A function that processes each frame and returns a list of inference results.</param>
        /// <returns>A dictionary where the key is the frame index and the value is a list of inference results of type <typeparamref name="T"/>.</returns>
        public Dictionary<int, List<T>> RunVideo<T>(
            VideoOptions options,
            double confidence,
            double iouThreshold,
            Func<SKImage, double, double, List<T>> func) where T : class, new()
        {
            var output = new Dictionary<int, List<T>>();

            using var _videoHandler = new VideoHandler.VideoHandler(options, useCuda);

            _videoHandler.ProgressEvent += (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _videoHandler.VideoCompleteEvent += (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _videoHandler.StatusChangeEvent += (sender, e) => VideoStatusEvent?.Invoke(sender, e);
            _videoHandler.FramesExtractedEvent += (sender, e) =>
            {
                output = RunBatchInferenceOnVideoFrames<T>(_videoHandler, confidence, iouThreshold, func);

                if (options.GenerateVideo)
                    _videoHandler.ProcessVideoPipeline(VideoAction.CompileFrames);
                else
                    VideoCompleteEvent?.Invoke(null, null!);
            };

            _videoHandler.ProcessVideoPipeline();

            return output;
        }

        /// <summary>
        /// Runs batch inference on the extracted video frames.
        /// </summary>
        private Dictionary<int, List<T>> RunBatchInferenceOnVideoFrames<T>(
            VideoHandler.VideoHandler _videoHandler,
            double confidence, double iouThreshold,
            Func<SKImage, double, double, List<T>> func) where T : class, new()
        {
            var frames = _videoHandler.GetExtractedFrames();
            int progressCounter = 0;
            int totalFrames = frames.Length;
            var batch = new List<T>[totalFrames];

            var shouldDrawLabelsOnKeptFrames = _videoHandler._videoSettings.DrawLabels && _videoHandler._videoSettings.KeepFrames;
            var shouldDrawLabelsOnVideoFrames = _videoHandler._videoSettings.DrawLabels && _videoHandler._videoSettings.GenerateVideo;

            _ = Parallel.For(0, totalFrames, parallelOptions, i =>
            {
                var frame = frames[i];
                using var img = SKImage.FromEncodedData(frame);


                var results = func.Invoke(img, confidence, iouThreshold);
                batch[i] = results;

                if (shouldDrawLabelsOnKeptFrames || shouldDrawLabelsOnVideoFrames)
                    DrawResultsOnVideoFrame(img, results, frame, _videoHandler._videoSettings);

                Interlocked.Increment(ref progressCounter);
                var progress = ((double)progressCounter / totalFrames) * 100;

                lock (_progressLock)
                {
                    VideoProgressEvent?.Invoke((int)progress, null!);
                }
            });

            return Enumerable.Range(0, batch.Length).ToDictionary(x => x, x => batch[x]);
        }

        /// <summary>
        /// Draw labels on video frames
        /// </summary>
        private static void DrawResultsOnVideoFrame<T>(SKImage img, List<T> results, string savePath, VideoSettings videoSettings)
        {
            var drawConfidence = videoSettings.DrawConfidence;

            using SKImage tmpImage = results switch
            {
                List<Classification> classifications => img.Draw(classifications, drawConfidence),
                List<ObjectDetection> objectDetections => img.Draw(objectDetections, drawConfidence),
                List<OBBDetection> obbDetections => img.Draw(obbDetections, drawConfidence),
                List<Segmentation> segmentations => img.Draw(segmentations, videoSettings.DrawSegment, drawConfidence),
                List<PoseEstimation> poseEstimations => img.Draw(poseEstimations, videoSettings.KeyPointOptions, drawConfidence),
                _ => throw new NotSupportedException("Unknown or incompatible ONNX model type."),
            };

            tmpImage.Save(savePath, SKEncodedImageFormat.Png, 100);
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

                if (result.Any(x => CalculateIoU(item.BoundingBox, x.BoundingBox) > iouThreshold) is false)
                    result.Add(item);
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
            var intersectionArea = CalculateArea(SKRectI.Intersect(a, b));

            return intersectionArea == 0
                ? 0
                : intersectionArea / (CalculateArea(a) + CalculateArea(b) - intersectionArea);
        }

        /// <summary>
        /// Calculates the padding and scaling factor needed to adjust the bounding box
        /// so that the detected object can be resized to match the original image size.
        /// </summary>
        /// <param name="image">The image for which the bounding box needs to be adjusted.</param>
        /// <param name="model">The ONNX model containing the input dimensions.</param>
        public (int, int, float) CalculateGain(SKImage image)
        {
            var model = OnnxModel;

            var (w, h) = (image.Width, image.Height);

            var gain = Math.Max((float)w / model.Input.Width, (float)h / model.Input.Height);
            var ratio = Math.Min(model.Input.Width / (float)image.Width, model.Input.Height / (float)image.Height);
            var (xPad, yPad) = ((int)(model.Input.Width - w * ratio) / 2, (int)(model.Input.Height - h * ratio) / 2);

            return (xPad, yPad, gain);
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
            if (_isDisposed) return;

            _session?.Dispose();
            _ortIoBinding?.Dispose();
            _runOptions?.Dispose();

            _isDisposed = true;
            GC.SuppressFinalize(this);
        }
    }
}
