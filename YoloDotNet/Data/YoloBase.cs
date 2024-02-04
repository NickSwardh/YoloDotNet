using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

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
        protected readonly ParallelOptions _parallelOptions;
        private readonly bool _useCuda;

        private readonly object _progressLock = new();

        #region Contracts
        public abstract List<Classification> RunClassification(Image img, int classes);
        public abstract List<ObjectDetection> RunObjectDetection(Image img, double threshold);
        public abstract List<Segmentation> RunSegmentation(Image img, double threshold);
        public abstract Dictionary<int, List<Classification>> RunClassification(VideoOptions options, int classes);
        public abstract Dictionary<int, List<ObjectDetection>> RunObjectDetection(VideoOptions options, double threshold);
        public abstract Dictionary<int, List<Segmentation>> RunSegmentation(VideoOptions options, double threshold);
        protected abstract List<Classification> ClassifyTensor(int numberOfClasses);
        protected abstract List<ObjectResult> ObjectDetectImage(Image image, double threshold);
        protected abstract List<Segmentation> SegmentImage(Image image, List<ObjectResult> boxes);
        #endregion

        protected Dictionary<string, Tensor<float>> Tensors { get; set; } = [];
        public OnnxModel OnnxModel { get; init; }

        /// <summary>
        /// Initializes a new instance of the Yolo base class.
        /// </summary>
        /// <param name="onnxModel">The path to the ONNX model file to use.</param>
        /// <param name="useCuda">Indicates whether to use CUDA for GPU acceleration.</param>
        /// <param name="gpuId">The GPU device ID to use when CUDA is enabled.</param>
        public YoloBase(string onnxModel, bool useCuda, int gpuId)
        {
            _session = useCuda
                ? new InferenceSession(onnxModel, SessionOptions.MakeSessionOptionWithCudaProvider(gpuId))
                : new InferenceSession(onnxModel);

            OnnxModel = _session.GetOnnxProperties();

            _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
            _useCuda = useCuda;
        }

        protected List<T> Run<T>(Image img, double limit, ModelType expectedModel)
        {
            VerifyExpectedModelType(expectedModel);

            using var resizedImg = img.ResizeImage(OnnxModel.Input.Width, OnnxModel.Input.Height);
            var tensorPixels = resizedImg.PixelsToTensor(OnnxModel.Input.BatchSize, OnnxModel.Input.Channels);

            lock (_progressLock)
            {
                using var result = _session.Run(new List<NamedOnnxValue>()
                {
                    NamedOnnxValue.CreateFromTensor(OnnxModel.InputName, tensorPixels),
                });

                Tensors = result.ToDictionary(x => x.Name, x => x.AsTensor<float>());

                return InvokeInferenceType(img, limit) is List<T> results
                    ? results
                    : [];
            }
        }

        /// <summary>
        /// Invokes inference based on the ONNX model type, processing the input tensor and image data.
        /// </summary>
        private object InvokeInferenceType(Image img, double limit) => OnnxModel.ModelType switch
        {
            ModelType.Classification => ClassifyTensor((int)limit),
            ModelType.ObjectDetection => ObjectDetectImage(img, limit).Select(x => (ObjectDetection)x).ToList(),
            ModelType.Segmentation => SegmentImage(img, ObjectDetectImage(img, limit)),
            _ => throw new NotSupportedException($"Unknown ONNX model")
        };

        /// <summary>
        /// Runs inference on video data using the specified options and optional confidence threshold.
        /// Trigger events for progress, completion, and status changes during video processing.
        /// </summary>
        /// <param name="options">Options for configuring video processing.</param>
        /// <param name="threshold">Optional. The confidence threshold for inference.</param>
        protected Dictionary<int, List<T>> RunVideo<T>(VideoOptions options, double threshold, ModelType expectedModel) where T : class, new()
        {
            VerifyExpectedModelType(expectedModel);

            var output = new Dictionary<int, List<T>>();

            using var _videoHandler = new VideoHandler.VideoHandler(options, _useCuda);

            _videoHandler.ProgressEvent += (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _videoHandler.VideoCompleteEvent += (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _videoHandler.StatusChangeEvent += (sender, e) => VideoStatusEvent?.Invoke(sender, e);
            _videoHandler.FramesExtractedEvent += (sender, e) =>
            {
                output = RunBatchInferenceOnVideoFrames<T>(_videoHandler, threshold);

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
        private Dictionary<int, List<T>> RunBatchInferenceOnVideoFrames<T>(VideoHandler.VideoHandler _videoHandler, double limit) where T : class, new()
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

                var results = Run<T>(img, limit, OnnxModel.ModelType);
                batch[i] = results;

                if (shouldDrawLabelsOnKeptFrames || shouldDrawLabelsOnVideoFrames)
                    DrawResultsOnVideoFrame(img, results, frame, _videoHandler._videoSettings.DrawConfidence);

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
        private static void DrawResultsOnVideoFrame<T>(Image<Rgba32> img, List<T> results, string savePath, bool drawConfidence)
        {
            if (results is List<Classification> classifications)
                img.Draw(classifications, drawConfidence);
            else if (results is List<ObjectDetection> objectDetections)
                img.Draw(objectDetections, drawConfidence);
            else if (results is List<Segmentation> segmentations)
                img.Draw(segmentations, drawConfidence);
            else
                throw new NotSupportedException("Unknown or incompatible ONNX model type.");

            img.SaveAsync(savePath).ConfigureAwait(false);
        }

        /// <summary>
        /// Removes overlapping bounding boxes in a list of object detection results.
        /// </summary>
        /// <param name="predictions">The list of object detection results to process.</param>
        /// <returns>A filtered list with non-overlapping bounding boxes based on confidence scores.</returns>
        protected static List<ObjectResult> RemoveOverlappingBoxes(List<ObjectResult> predictions)
        {
            var result = new List<ObjectResult>();

            foreach (var item in predictions)
            {
                var rect1 = item.BoundingBox;
                var overlappingItem = result.FirstOrDefault(current =>
                {
                    var rect2 = current.BoundingBox;
                    var intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Width * intersection.Height; // intersection area
                    float unionArea = rect1.Width * rect1.Height + rect2.Width * rect2.Height - intArea; // union area
                    float overlap = intArea / unionArea; // overlap ratio

                    return overlap > 0.45f;
                });

                if (overlappingItem is not null)
                {
                    if (item.Confidence >= overlappingItem.Confidence)
                    {
                        result.Remove(overlappingItem);
                        result.Add(item); // Replace the current overlapping box with the higher confidence one
                    }
                }
                else
                    result.Add(item);
            }

            return result;
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
        /// Calculate
        /// </summary>
        protected static float CalculatePixelConfidence(byte value) => 1 - value / 255F;

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
            _session.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
