using System.Buffers;
using YoloDotNet.Extensions;

namespace YoloDotNet.Data
{
    /// <summary>
    /// Abstract base class for performing image vision tasks using a YOLOv8 model in ONNX format.
    /// </summary>
    public abstract class YoloBase : IDisposable
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

        private readonly InferenceSession _session;
        private readonly RunOptions _runOptions;
        private readonly OrtIoBinding _ortIoBinding;
        protected readonly ParallelOptions _parallelOptions;
        private readonly bool _useCuda;

        private int tensorBufferSize;
        private ArrayPool<float> customSizeFloatPool;

        private readonly object _progressLock = new();

        #region Contracts
        public abstract List<Classification> RunClassification(Image img, int classes);
        public abstract List<ObjectDetection> RunObjectDetection(Image img, double confidence, double iou);
        public abstract List<Segmentation> RunSegmentation(Image img, double confidence, double iou);
        public abstract List<PoseEstimation> RunPoseEstimation(Image img, double confidence, double iou);
        public abstract List<OBBDetection> RunObbDetection(Image img, double confidence, double iou);
        public abstract Dictionary<int, List<Classification>> RunClassification(VideoOptions options, int classes);
        public abstract Dictionary<int, List<ObjectDetection>> RunObjectDetection(VideoOptions options, double confidence, double iou);
        public abstract Dictionary<int, List<OBBDetection>> RunObbDetection(VideoOptions options, double confidence, double iou);
        public abstract Dictionary<int, List<Segmentation>> RunSegmentation(VideoOptions options, double confidence, double iou);
        public abstract Dictionary<int, List<PoseEstimation>> RunPoseEstimation(VideoOptions options, double confidence, double iou);
        protected abstract List<Classification> ClassifyTensor(int numberOfClasses);
        protected abstract ObjectResult[] ObjectDetectImage(Image image, double confidence, double iou);
        protected abstract List<Segmentation> SegmentImage(Image image, ObjectResult[] boxes);
        protected abstract List<PoseEstimation> PoseEstimateImage(Image image, double confidence, double iou);
        #endregion

        protected Dictionary<string, OrtValue> Tensors { get; set; } = [];

        public OnnxModel OnnxModel { get; init; }

        /// <summary>
        /// Initializes a new instance of the Yolo base class.
        /// </summary>
        /// <param name="onnxModel">The path to the ONNX model file to use.</param>
        /// <param name="useCuda">Indicates whether to use CUDA for GPU acceleration.</param>
        /// <param name="gpuId">The GPU device ID to use when CUDA is enabled.</param>
        public YoloBase(string onnxModel, bool useCuda, bool allocateGpuMemory, int gpuId)
        {
            _useCuda = useCuda;
            _session = useCuda
                ? new InferenceSession(onnxModel, SessionOptions.MakeSessionOptionWithCudaProvider(gpuId))
                : new InferenceSession(onnxModel);

            _runOptions = new RunOptions();
            _ortIoBinding = _session.CreateIoBinding();

            OnnxModel = _session.GetOnnxProperties();

            _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

            if (useCuda && allocateGpuMemory)
                _session.AllocateGpuMemory(_ortIoBinding, _runOptions);

            // tensorBufferSize can be calculated once and reused for all calls, as it is based on the model properties
            this.tensorBufferSize = OnnxModel.Input.BatchSize * OnnxModel.Input.Channels * OnnxModel.Input.Width * OnnxModel.Input.Height;
            this.customSizeFloatPool = ArrayPool<float>.Create(maxArrayLength: this.tensorBufferSize + 1, maxArraysPerBucket: 10);
        }

        /// <summary>
        /// Run inference on image
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="img"></param>
        /// <param name="confidence"></param>
        /// <param name="iouThreshold"></param>
        /// <param name="expectedModel"></param>
        protected List<T> Run<T>(Image img, double confidence, double iouThreshold, ModelType expectedModel, bool preloadGpu = false)
        {
            VerifyExpectedModelType(expectedModel);

            using var resizedImg = img.ResizeImage(OnnxModel.Input.Width, OnnxModel.Input.Height);

            var tensorArrayBuffer = this.customSizeFloatPool.Rent(minimumLength: tensorBufferSize);

            try
            {
                lock (_progressLock)
                {
                    var tensorPixels = resizedImg.NormalizePixelsToTensor(
                        inputBatchSize: OnnxModel.Input.BatchSize,
                        inputChannels: OnnxModel.Input.Channels,
                        tensorBufferSize: this.tensorBufferSize,
                        tensorArrayBuffer: tensorArrayBuffer);

                    using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensorPixels.Buffer, OnnxModel.InputShape);

                    var inputNames = new Dictionary<string, OrtValue>
                {
                    { OnnxModel.InputName, inputOrtValue }
                };

                    using var ortResults = _session.Run(new RunOptions(), inputNames, OnnxModel.OutputNames);

                    Tensors = OnnxModel.OutputNames.Zip(ortResults, (output, ort) => new { output, ort })
                        .ToDictionary(item => item.output, item => item.ort);

                    return InvokeInferenceType(img, confidence, iouThreshold) is List<T> results
                        ? results
                        : [];
                }
            }
            finally
            {
                this.customSizeFloatPool.Return(tensorArrayBuffer);
            }
        }

        /// <summary>
        /// Invokes inference based on the ONNX model type, processing the input tensor and image data.
        /// </summary>
        private object InvokeInferenceType(Image img, double confidence, double iouThreshold) => OnnxModel.ModelType switch
        {
            ModelType.Classification => ClassifyTensor((int)confidence),
            ModelType.ObjectDetection => ObjectDetectImage(img, confidence, iouThreshold).Select(x => (ObjectDetection)x).ToList(),
            ModelType.Segmentation => SegmentImage(img, ObjectDetectImage(img, confidence, iouThreshold)),
            ModelType.PoseEstimation => PoseEstimateImage(img, confidence, iouThreshold),
            ModelType.ObbDetection => ObjectDetectImage(img, confidence, iouThreshold).Select(x => (OBBDetection)x).ToList(),
            _ => throw new NotSupportedException($"Unknown ONNX model")
        };

        /// <summary>
        /// Runs inference on video data using the specified options and optional confidence threshold.
        /// Trigger events for progress, completion, and status changes during video processing.
        /// </summary>
        /// <param name="options">Options for configuring video processing.</param>
        /// <param name="confidence">Confidence threshold for inference.</param>
        /// <param name="iouThreshold">IoU threshold value for excluding bounding boxes</param>
        /// <param name="expectedModel">Onnx modeltype</param>
        protected Dictionary<int, List<T>> RunVideo<T>(VideoOptions options, double confidence, double iouThreshold, ModelType expectedModel) where T : class, new()
        {
            VerifyExpectedModelType(expectedModel);

            var output = new Dictionary<int, List<T>>();

            using var _videoHandler = new VideoHandler.VideoHandler(options, _useCuda);

            _videoHandler.ProgressEvent += (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _videoHandler.VideoCompleteEvent += (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _videoHandler.StatusChangeEvent += (sender, e) => VideoStatusEvent?.Invoke(sender, e);
            _videoHandler.FramesExtractedEvent += (sender, e) =>
            {
                output = RunBatchInferenceOnVideoFrames<T>(_videoHandler, confidence, iouThreshold);

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
        private Dictionary<int, List<T>> RunBatchInferenceOnVideoFrames<T>(VideoHandler.VideoHandler _videoHandler, double confidence, double iouThreshold) where T : class, new()
        {
            var frames = _videoHandler.GetExtractedFrames();
            int progressCounter = 0;
            int totalFrames = frames.Length;
            var batch = new List<T>[totalFrames];

            var shouldDrawLabelsOnKeptFrames = _videoHandler._videoSettings.DrawLabels && _videoHandler._videoSettings.KeepFrames;
            var shouldDrawLabelsOnVideoFrames = _videoHandler._videoSettings.DrawLabels && _videoHandler._videoSettings.GenerateVideo;

            _ = Parallel.For(0, totalFrames, _parallelOptions, i =>
            {
                var frame = frames[i];
                using var img = Image.Load<Rgba32>(frame);

                var results = Run<T>(img, confidence, iouThreshold, OnnxModel.ModelType);
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
        private static void DrawResultsOnVideoFrame<T>(Image<Rgba32> img, List<T> results, string savePath, VideoSettings videoSettings)
        {
            var drawConfidence = videoSettings.DrawConfidence;

            switch (results)
            {
                case List<Classification> classifications:
                    img.Draw(classifications, drawConfidence);
                    break;
                case List<ObjectDetection> objectDetections:
                    img.Draw(objectDetections, drawConfidence);
                    break;
                case List<OBBDetection> obbDetections:
                    img.Draw(obbDetections, drawConfidence);
                    break;
                case List<Segmentation> segmentations:
                    img.Draw(segmentations, videoSettings.DrawSegment, drawConfidence);
                    break;
                case List<PoseEstimation> poseEstimations:
                    img.Draw(poseEstimations, videoSettings.PoseOptions, drawConfidence);
                    break;
                default:
                    throw new NotSupportedException("Unknown or incompatible ONNX model type.");
            }

            img.SaveAsync(savePath).ConfigureAwait(false);
        }

        /// <summary>
        /// Removes overlapping bounding boxes in a list of object detection results.
        /// </summary>
        /// <param name="predictions">The list of object detection results to process.</param>
        /// <param name="iouThreshold">Higher Iou-threshold result in fewer detections by excluding overlapping boxes.</param>
        /// <returns>A filtered list with non-overlapping bounding boxes based on confidence scores.</returns>
        protected static ObjectResult[] RemoveOverlappingBoxes(ObjectResult[] predictions, double iouThreshold)
        {
            Array.Sort(predictions, (a, b) => b.Confidence.CompareTo(a.Confidence));
            var result = new HashSet<ObjectResult>();

            var span = predictions.AsSpan();

            for (int i = 0; i < span.Length; i++)
            {
                var item = span[i];

                if (result.Any(x => CalculateIoU(item.BoundingBox, x.BoundingBox) > iouThreshold) is false)
                    result.Add(item);
            }

            return [.. result];
        }

        /// <summary>
        /// Squash value to a number between 0 and 1
        /// </summary>
        protected static float Sigmoid(float value) => 1 / (1 + MathF.Exp(-value));

        /// <summary>
        /// Calculate pixel luminance
        /// </summary>
        protected static byte CalculatePixelLuminance(float value) => (byte)(255 - value * 255);

        /// <summary>
        /// Calculate pixel by byte to confidence
        /// </summary>
        protected static float CalculatePixelConfidence(byte value) => 1 - value / 255F;

        /// <summary>
        /// Calculate radian to degree
        /// </summary>
        /// <param name="value"></param>
        protected static float CalculateRadianToDegree(float value) => value * (180 / (float)Math.PI);

        /// <summary>
        /// Calculate rectangle area
        /// </summary>
        /// <param name="rect"></param>
        protected static float CalculateArea(RectangleF rect) => rect.Width * rect.Height;

        /// <summary>
        /// Calculate IoU (Intersection Over Union) bounding box overlap.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        protected static float CalculateIoU(RectangleF a, RectangleF b)
        {
            var intersectionArea = CalculateArea(RectangleF.Intersect(a, b));

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
        protected static (int, int, float) CalculateGain(Image image, OnnxModel model)
        {
            var (w, h) = (image.Width, image.Height);

            var gain = Math.Max((float)w / model.Input.Width, (float)h / model.Input.Height);
            var ratio = Math.Min(model.Input.Width / (float)image.Width, model.Input.Height / (float)image.Height);
            var (xPad, yPad) = ((int)(model.Input.Width - w * ratio) / 2, (int)(model.Input.Height - h * ratio) / 2);

            return (xPad, yPad, gain);
        }

        /// <summary>
        /// Verify that loaded model is of the expected type
        /// </summary>
        private void VerifyExpectedModelType(ModelType expectedModelType)
        {
            if (expectedModelType.Equals(OnnxModel.ModelType) is false)
                throw new Exception($"Loaded ONNX-model is of type {OnnxModel.ModelType} and can't be used for {expectedModelType}.");
        }

        /// <summary>
        /// Releases resources and suppresses the finalizer for the current object.
        /// </summary>
        public void Dispose()
        {
            _ortIoBinding?.Dispose();
            _runOptions.Dispose();
            _session.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
