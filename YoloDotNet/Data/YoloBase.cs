using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections.Concurrent;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

namespace YoloDotNet.Data
{
    /// <summary>
    /// Abstract base class for performing object detection using a YOLOv8 model in ONNX format.
    /// </summary>
    public abstract class YoloBase : IDisposable
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

        private readonly InferenceSession _session;
        private readonly ParallelOptions _parallelOptions;
        private readonly bool _useCuda;

        private readonly object _progressLock = new();

        public abstract List<Classification> ClassifyImage(int numberOfClasses);
        public abstract List<ObjectResult> ObjectDetectImage(Image image, double threshold);
        public abstract List<Segmentation> SegmentImage(Image image, List<ObjectResult> boxes);

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

        /// <summary>
        /// Run image classification on an Image.
        /// </summary>
        /// <param name="img">The image to classify.</param>
        /// <param name="classes">The number of classes to return (default is 1).</param>
        /// <returns>A list of classification results.</returns>
        public List<Classification> RunClassification(Image img, int classes = 1)
            => Run<Classification>(img, classes, ModelType.Classification);

        /// <summary>
        /// Run object detection on an image.
        /// </summary>
        /// <param name="img">The image for object detection.</param>
        /// <param name="threshold">The confidence threshold for detected objects (default is 0.25).</param>
        /// <returns>A list of detected objects.</returns>
        public List<ObjectDetection> RunObjectDetection(Image img, double threshold = 0.25)
            => Run<ObjectDetection>(img, threshold, ModelType.ObjectDetection);

        /// <summary>
        /// Run image classification on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="classes">The number of classes to return for each frame (default is 1).</param>
        public void RunClassification(VideoOptions options, int classes = 1)
            => RunVideo(options, classes, ModelType.Classification);

        /// <summary>
        /// Run object detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="threshold">The confidence threshold for detected objects (default is 0.25).</param>
        public void RunObjectDetection(VideoOptions options, double threshold = 0.25)
            => RunVideo(options, threshold, ModelType.ObjectDetection);

        private List<T> Run<T>(Image img, double limit, ModelType modelType)
        {
            if (OnnxModel.ModelType != modelType)
                ThrowInvalidOperationException(modelType);

            using var resizedImg = img.ResizeImage(OnnxModel.Input.Width, OnnxModel.Input.Height);

            var tensorPixels = resizedImg.ExtractPixelsFromImage(OnnxModel.Input.BatchSize, OnnxModel.Input.Channels);

            lock (_progressLock)
            {
                using var result = _session.Run(new List<NamedOnnxValue>()
                {
                    NamedOnnxValue.CreateFromTensor(OnnxModel.InputName, tensorPixels)
                });

                var tensor = result.FirstOrDefault(x => x.Name == OnnxModel.OutputName)!.AsTensor<float>();

                return InvokeInferenceType(tensor, img, limit) is List<T> results
                    ? results
                    : [];
            }
        }

        /// <summary>
        /// Invokes inference based on the ONNX model type, processing the input tensor and image data.
        /// </summary>
        private object InvokeInferenceType(Tensor<float> tensor, Image img, double limit) => OnnxModel.ModelType switch
        {
            ModelType.Classification => ClassifyImage((int)limit),
            ModelType.ObjectDetection => ObjectDetectImage(img, limit).Select(x => (ObjectDetection)x).ToList(),
            _ => throw new NotSupportedException($"Unknown ONNX model")
        };

        /// <summary>
        /// Runs object detection inference on an input image.
        /// </summary>
        /// <param name="img">The input image to perform object detection on.</param>
        /// <param name="threshold">Optional. The confidence threshold for accepting object detections.</param>
        /// <returns>A list of result models representing detected objects.</returns>
        private List<ObjectDetection> ObjectDetectImage(Tensor<float> tensor, Image img, double threshold)
            => [.. RemoveOverlappingBoxes(DetectObjectsInTensor(tensor, img, threshold))];

        /// <summary>
        /// Runs inference on video data using the specified options and optional confidence threshold.
        /// Trigger events for progress, completion, and status changes during video processing.
        /// </summary>
        /// <param name="options">Options for configuring video processing.</param>
        /// <param name="threshold">Optional. The confidence threshold for inference.</param>
        private void RunVideo(VideoOptions options, double threshold, ModelType modelType)
        {
            if (OnnxModel.ModelType != modelType)
                ThrowInvalidOperationException(modelType);

            using var _videoHandler = new VideoHandler.VideoHandler(options, _useCuda);

            _videoHandler.ProgressEvent += (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _videoHandler.VideoCompleteEvent += (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _videoHandler.StatusChangeEvent += (sender, e) => VideoStatusEvent?.Invoke(sender, e);
            _videoHandler.FramesExtractedEvent += (sender, e) =>
            {
                RunBatchInferenceOnVideoFrames(_videoHandler, options, threshold);
                _videoHandler.ProcessVideoPipeline(VideoAction.CompileFrames);
            };

            _videoHandler.ProcessVideoPipeline();
        }

        /// <summary>
        /// Runs batch inference on the extracted video frames.
        /// </summary>
        private void RunBatchInferenceOnVideoFrames(VideoHandler.VideoHandler _videoHandler, VideoOptions options, double limit)
        {
            var frames = new ConcurrentBag<string>(_videoHandler.GetExtractedFrames());
            int progressCounter = 0;
            int totalFrames = frames.Count;

            Parallel.ForEach(frames, _parallelOptions, (frame, token) =>
            {
                using var img = Image.Load<Rgba32>(frame);

                switch (OnnxModel.ModelType)
                {
                    case ModelType.Classification:
                        img.DrawClassificationLabels(RunClassification(img, (int)limit), options.DrawConfidence);
                        break;
                    case ModelType.ObjectDetection:
                        img.DrawBoundingBoxes(RunObjectDetection(img, limit), options.DrawConfidence);
                        break;
                    default: throw new NotSupportedException("Unknown ONNX model.");
                };

                img.Save(frame);

                Interlocked.Increment(ref progressCounter);
                var progress = ((double)progressCounter / totalFrames) * 100;

                lock (_progressLock)
                {
                    VideoProgressEvent?.Invoke((int)progress, null!);
                }
            });
        }

        /// <summary>
        /// Removes overlapping bounding boxes in a list of object detection results.
        /// </summary>
        /// <param name="predictions">The list of object detection results to process.</param>
        /// <returns>A filtered list with non-overlapping bounding boxes based on confidence scores.</returns>
        private static List<ObjectDetection> RemoveOverlappingBoxes(List<ObjectDetection> predictions)
        {
            var result = new List<ObjectDetection>();

            foreach (var item in predictions)
            {
                var rect1 = item.Rectangle;
                var overlappingItem = result.FirstOrDefault(current =>
                {
                    var rect2 = current.Rectangle;
                    var intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Width * intersection.Height; // intersection area
                    float unionArea = rect1.Width * rect1.Height + rect2.Width * rect2.Height - intArea; // union area
                    float overlap = intArea / unionArea; // overlap ratio

                    return overlap >= 0.45f;
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

        private void ThrowInvalidOperationException(ModelType expectedModellType)
            => throw new InvalidOperationException($"Loaded ONNX-model is of type {OnnxModel.ModelType} and can't be used for {expectedModellType}.");

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
        /// Releases resources and suppresses the finalizer for the current object.
        /// </summary>
        public void Dispose()
        {
            _session.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
