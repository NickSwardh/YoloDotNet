using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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

        private object _progressLock = new();

        public abstract List<ResultModel> DetectObjectsInTensor(Tensor<float> tensor, Image image, double threshold);

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
        /// Runs object detection inference on an input image.
        /// </summary>
        /// <param name="img">The input image to perform object detection on.</param>
        /// <param name="threshold">Optional. The confidence threshold for accepting object detections (default is 0.25).</param>
        /// <returns>A list of result models representing detected objects.</returns>
        public List<ResultModel> RunInference(Image img, double threshold = 0.25)
        {
            using var resizedImg = img.ResizeImage(OnnxModel.Input.Width, OnnxModel.Input.Height);

            var tensorPixels = resizedImg.ExtractPixelsFromImage(OnnxModel.Input.BatchSize, OnnxModel.Input.Channels);

            lock (_progressLock)
            {
                using var result = _session.Run(new List<NamedOnnxValue>()
                {
                    NamedOnnxValue.CreateFromTensor(OnnxModel.InputName, tensorPixels)
                });

                var tensor = result.FirstOrDefault(x => x.Name == OnnxModel.OutputName)!.AsTensor<float>();

                return RemoveOverlappingBoxes(DetectObjectsInTensor(tensor, img, threshold));
            }
        }

        /// <summary>
        /// Runs inference on video data using the specified options and optional confidence threshold.
        /// Trigger events for progress, completion, and status changes during video processing.
        /// </summary>
        /// <param name="options">Options for configuring video processing.</param>
        /// <param name="threshold">Optional. The confidence threshold for inference (default is 0.25).</param>
        public void RunInference(VideoOptions options, double threshold = 0.25)
        {
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
        private void RunBatchInferenceOnVideoFrames(VideoHandler.VideoHandler _videoHandler, VideoOptions options, double threshold)
        {
            var frames = new ConcurrentBag<string>(_videoHandler.GetExtractedFrames());
            int progressCounter = 0;
            int totalFrames = frames.Count;

            Parallel.ForEach(frames, _parallelOptions, (frame, token) =>
            {
                using var img = Image.Load<Rgba32>(frame);

                img.DrawBoundingBoxes(RunInference(img, threshold), options.DrawConfidence);
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
        public static List<ResultModel> RemoveOverlappingBoxes(List<ResultModel> predictions)
        {
            var result = new List<ResultModel>();

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
