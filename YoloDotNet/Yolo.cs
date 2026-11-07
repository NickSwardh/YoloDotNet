// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet
{
    /// <summary>
    /// Initializes a new instance of YoloDotNet.
    /// </summary>
    /// <param name="options">Options for initializing the YoloDotNet model.</param>
    public class Yolo(YoloOptions options) : IDisposable
    {
        #region Private fields
        private readonly IModule _detection = ModuleFactory.CreateModule(options);
        private FFmpegService _ffmpegService = default!;
        #endregion

        #region Public Fields
        public OnnxModel OnnxModel => _detection.OnnxModel;
        /// <summary>
        /// Called for every decoded frame, with the frame and its zero-based index, after <see cref="StartVideoProcessing()"/>.
        /// Run detection yourself in it; drawing on the frame is kept in the output video. Frames are handled one after the other.
        /// </summary>
        /// <remarks>
        /// For faster processing, use <see cref="StartVideoProcessing{T}(Action{SKBitmap, List{T}, long}, double, double, SKRectI?, double, int, int)"/>
        /// instead: it runs the loaded model itself and detects upcoming frames while your handler processes the current one.
        /// Keep using this callback when each frame needs custom detection, such as a different ROI per frame, several models, or skipping frames.
        /// </remarks>
        public Action<SKBitmap, long> OnVideoFrameReceived = default!;
        public Action OnVideoEnd = default!;
        #endregion

        #region Classification

        /// <summary>
        /// Run image classification on an Image.
        /// </summary>
        /// <param name="img">The SKBitmap to classify.</param>
        /// <param name="classes">The number of classes to return (default is 1).</param>
        /// <returns>A list of classification results.</returns>
        public List<Classification> RunClassification(SKBitmap img, int classes = 1)
            => ((IClassificationModule)_detection).ProcessImage(img, classes, 0, 0);

        /// <summary>
        /// Run image classification on an Image.
        /// </summary>
        /// <param name="img">The SKImage to classify.</param>
        /// <param name="classes">The number of classes to return (default is 1).</param>
        /// <returns>A list of classification results.</returns>
        public List<Classification> RunClassification(SKImage img, int classes = 1)
            => ((IClassificationModule)_detection).ProcessImage(img, classes, 0, 0);

        #endregion

        #region Object Detection

        /// <summary>
        /// Run object detection on an Image.
        /// </summary>
        /// <param name="img">The SKBitmap to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <param name="roi">Optional region of interest within the image to focus detection on.</param>
        /// <returns>A list of classification results.</returns>
        public List<ObjectDetection> RunObjectDetection(SKBitmap img, double confidence = 0.2, double iou = 0.7, SKRectI? roi = null)
            => ((IObjectDetectionModule)_detection).ProcessImage(img, confidence, 0, iou, roi);

        /// <summary>
        /// Run object detection on an Image.
        /// </summary>
        /// <param name="img">The SKImage to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <param name="roi">Optional region of interest within the image to focus detection on.</param>
        /// <returns>A list of classification results.</returns>
        public List<ObjectDetection> RunObjectDetection(SKImage img, double confidence = 0.2, double iou = 0.7, SKRectI? roi = null)
             => ((IObjectDetectionModule)_detection).ProcessImage(img, confidence, 0, iou, roi);

        #endregion

        #region OBB (Oriented Bounding Box)

        /// <summary>
        /// Run oriented bounding bBox detection on an image.
        /// </summary>
        /// <param name="img">The SKBitmap to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <param name="roi">Optional region of interest within the image to focus detection on.</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<OBBDetection> RunObbDetection(SKBitmap img, double confidence = 0.2, double iou = 0.7, SKRectI? roi = null)
            => ((IOBBDetectionModule)_detection).ProcessImage(img, confidence, 0, iou, roi);

        /// <summary>
        /// Run oriented bounding bBox detection on an image.
        /// </summary>
        /// <param name="img">The SKImage to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <param name="roi">Optional region of interest within the image to focus detection on.</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<OBBDetection> RunObbDetection(SKImage img, double confidence = 0.2, double iou = 0.7, SKRectI? roi = null)
            => ((IOBBDetectionModule)_detection).ProcessImage(img, confidence, 0, iou, roi);

        #endregion

        #region Segmentation

        /// <summary>
        /// Run segmentation on an image.
        /// </summary>
        /// <param name="img">The SKBitmap to segmentate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="pixelConfedence">The confidence threshold for a pixel to belong to a segmentation mask (default: 0.65).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <param name="roi">Optional region of interest within the image to focus detection on.</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<Segmentation> RunSegmentation(SKBitmap img, double confidence = 0.2, double pixelConfedence = 0.65, double iou = 0.7, SKRectI? roi = null)
            => ((ISegmentationModule)_detection).ProcessImage(img, confidence, pixelConfedence, iou, roi);

        /// <summary>
        /// Run segmentation on an image.
        /// </summary>
        /// <param name="img">The SKImage to segmentate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="pixelConfedence">The confidence threshold for a pixel to belong to a segmentation mask (default: 0.65).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <param name="roi">Optional region of interest within the image to focus detection on.</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<Segmentation> RunSegmentation(SKImage img, double confidence = 0.2, double pixelConfedence = 0.65, double iou = 0.7, SKRectI? roi = null)
            => ((ISegmentationModule)_detection).ProcessImage(img, confidence, pixelConfedence, iou, roi);

        #endregion

        #region Pose Estimation

        /// <summary>
        /// Run pose estimation on an image.
        /// </summary>
        /// <param name="img">The SKBitmap to pose estimate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <param name="roi">Optional region of interest within the image to focus detection on.</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<PoseEstimation> RunPoseEstimation(SKBitmap img, double confidence = 0.2, double iou = 0.7, SKRectI? roi = null)
            => ((IPoseEstimationModule)_detection).ProcessImage(img, confidence, 0, iou, roi);

        /// <summary>
        /// Run pose estimation on an image.
        /// </summary>
        /// <param name="img">The SKImage to pose estimate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <param name="roi">Optional region of interest within the image to focus detection on.</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<PoseEstimation> RunPoseEstimation(SKImage img, double confidence = 0.2, double iou = 0.7, SKRectI? roi = null)
            => ((IPoseEstimationModule)_detection).ProcessImage(img, confidence, 0, iou, roi);

        #endregion

        #region Video

        /// <summary>
        /// Initializes the video stream using the specified <see cref="VideoOptions"/> and sets up event handlers for frame processing and video completion.
        /// </summary>
        /// <param name="videoOptions"></param>
        public void InitializeVideo(VideoOptions videoOptions)
        {
            _ffmpegService = new(videoOptions, options)
            {
                OnFrameReady = (frame, frameIndex) => OnVideoFrameReceived.Invoke(frame, frameIndex),
                OnVideoEnd = () => OnVideoEnd?.Invoke()
            };
        }

        /// <summary>
        /// Retrieves a list of available video input devices detected on the current system.
        /// </summary>
        /// <exception cref="YoloDotNetVideoException"></exception>
        public static List<string> GetVideoDevices()
            => FFmpegService.GetVideoDevicesOnSystem() ?? throw new YoloDotNetVideoException(
                "No video initialized. Please call InitializeVideo() before attempting to retrieve metadata.");

        /// <summary>
        /// Retrieves metadata about the stream or initialized video, such as duration, frame rate, and resolution.
        /// </summary>
        /// <exception cref="YoloDotNetVideoException"></exception>
        public VideoMetadata GetVideoMetaData()
            => _ffmpegService.VideoMetadata ?? throw new YoloDotNetVideoException(
                "No video initialized. Please call InitializeVideo() before attempting to retrieve metadata.");

        /// <summary>
        /// Starts decoding and processing video frames from the initialized video stream. Every frame is passed to
        /// <see cref="OnVideoFrameReceived"/>, where you run detection yourself, one frame after the other.
        /// </summary>
        /// <remarks>
        /// For faster processing, use <see cref="StartVideoProcessing{T}(Action{SKBitmap, List{T}, long}, double, double, SKRectI?, double, int, int)"/>
        /// instead: it runs the loaded model itself and detects upcoming frames while your handler processes the current one.
        /// Use this method when each frame needs custom detection, such as a different ROI per frame, several models, or skipping frames.
        /// </remarks>
        public void StartVideoProcessing()
            => _ffmpegService.Start();

        /// <summary>
        /// Starts decoding the initialized video and runs the loaded model on every frame, detecting upcoming frames while
        /// <paramref name="onResult"/> handles the current one. Faster than <see cref="OnVideoFrameReceived"/> with
        /// <see cref="StartVideoProcessing()"/> when frames are available faster than they are processed, such as video files.
        /// </summary>
        /// <remarks>
        /// Alternative: assign <see cref="OnVideoFrameReceived"/>, run detection yourself in it, and call <see cref="StartVideoProcessing()"/>.
        /// Each frame is then detected and handled one after the other. Use that when each frame needs custom detection,
        /// such as a different ROI per frame, several models, or skipping frames.
        /// </remarks>
        /// <example>
        /// <code>
        /// yolo.InitializeVideo(videoOptions);
        ///
        /// yolo.StartVideoProcessing&lt;ObjectDetection&gt;((frame, result, frameIndex) =>
        /// {
        ///     result = result.Track(tracker);
        ///     frame.Draw(result);
        /// },
        /// confidence: 0.25, iou: 0.7);
        /// </code>
        /// </example>
        /// <typeparam name="T">The result type of the loaded model's task, for example <see cref="ObjectDetection"/> for a detection model.</typeparam>
        /// <param name="onResult">
        /// Called with every frame, its results and its index: on the calling thread, one frame at a time and in frame order,
        /// so tracking and other per-frame state work as with <see cref="OnVideoFrameReceived"/>. Drawing on the frame is kept in the output video.
        /// </param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2). Not used for classification.</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7). Not used for classification.</param>
        /// <param name="roi">Optional region of interest within each frame to focus detection on. Not used for classification.</param>
        /// <param name="pixelConfidence">Segmentation only: the confidence threshold for mask pixels (default is 0.65).</param>
        /// <param name="classes">Classification only: the number of classes to return (default is 1).</param>
        /// <param name="framesInFlight">The maximum number of frames detected at once (default is 2).</param>
        /// <exception cref="YoloDotNetVideoException">No video is initialized, or <typeparamref name="T"/> does not match the loaded model's task.</exception>
        public void StartVideoProcessing<T>(Action<SKBitmap, List<T>, long> onResult,
            double confidence = 0.2,
            double iou = 0.7,
            SKRectI? roi = null,
            double pixelConfidence = 0.65,
            int classes = 1,
            int framesInFlight = 2)
        {
            ArgumentNullException.ThrowIfNull(onResult);
            ArgumentOutOfRangeException.ThrowIfLessThan(framesInFlight, 1);

            if (_ffmpegService is null)
                throw new YoloDotNetVideoException("No video initialized. Please call InitializeVideo() before starting video processing.");

            // The loaded model decides the task; T only states the result type that onResult is compiled against.
            (Type ResultType, Func<SKBitmap, object> Detect) task = OnnxModel.ModelType switch
            {
                ModelType.Classification => (typeof(Classification), frame => RunClassification(frame, classes)),
                ModelType.ObjectDetection => (typeof(ObjectDetection), frame => RunObjectDetection(frame, confidence, iou, roi)),
                ModelType.ObbDetection => (typeof(OBBDetection), frame => RunObbDetection(frame, confidence, iou, roi)),
                ModelType.Segmentation => (typeof(Segmentation), frame => RunSegmentation(frame, confidence, pixelConfidence, iou, roi)),
                ModelType.PoseEstimation => (typeof(PoseEstimation), frame => RunPoseEstimation(frame, confidence, iou, roi)),
                _ => throw new YoloDotNetVideoException($"Video processing does not support model type {OnnxModel.ModelType}.")
            };

            if (task.ResultType != typeof(T))
            {
                throw new YoloDotNetVideoException($"The loaded model ({ModelInfo}) returns {task.ResultType.Name} results, " +
                    $"but video processing was started for {typeof(T).Name}. Use StartVideoProcessing<{task.ResultType.Name}>(...).");
            }

            _ffmpegService.Start(frame => (List<T>)task.Detect(frame), onResult, framesInFlight);
        }

        /// <summary>
        /// Stops video frame processing and releases resources associated with the video stream.
        /// </summary>
        public void StopVideoProcessing()
            => _ffmpegService.Stop();

        #endregion

        #region Model Info

        /// <summary>
        /// Gets a description of the currently loaded YOLO model,
        /// including the model type and version. Returns "No model loaded"
        /// if no model has been initialized.
        /// </summary>
        public string ModelInfo =>
            _detection.OnnxModel == null
                ? "No model loaded"
                : $"{_detection.OnnxModel.ModelType} (yolo {_detection.OnnxModel.ModelVersion.ToString().ToLower()})";

        #endregion

        #region Dispose

        public void Dispose()
        {
            _detection.Dispose();
            _ffmpegService?.Dispose();

            options.ExecutionProvider?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
