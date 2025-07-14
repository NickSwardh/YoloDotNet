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
        /// <returns>A list of classification results.</returns>
        public List<ObjectDetection> RunObjectDetection(SKBitmap img, double confidence = 0.2, double iou = 0.7)
            => ((IObjectDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);

        /// <summary>
        /// Run object detection on an Image.
        /// </summary>
        /// <param name="img">The SKImage to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of classification results.</returns>
        public List<ObjectDetection> RunObjectDetection(SKImage img, double confidence = 0.2, double iou = 0.7)
             => ((IObjectDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);

        #endregion

        #region OBB (Oriented Bounding Box)

        /// <summary>
        /// Run oriented bounding bBox detection on an image.
        /// </summary>
        /// <param name="img">The SKBitmap to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<OBBDetection> RunObbDetection(SKBitmap img, double confidence = 0.2, double iou = 0.7)
            => ((IOBBDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);

        /// <summary>
        /// Run oriented bounding bBox detection on an image.
        /// </summary>
        /// <param name="img">The SKImage to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<OBBDetection> RunObbDetection(SKImage img, double confidence = 0.2, double iou = 0.7)
            => ((IOBBDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);

        #endregion

        #region Segmentation

        /// <summary>
        /// Run segmentation on an image.
        /// </summary>
        /// <param name="img">The SKBitmap to segmentate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<Segmentation> RunSegmentation(SKBitmap img, double confidence = 0.2, double pixelConfedence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_detection).ProcessImage(img, confidence, pixelConfedence, iou);

        /// <summary>
        /// Run segmentation on an image.
        /// </summary>
        /// <param name="img">The SKImage to segmentate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<Segmentation> RunSegmentation(SKImage img, double confidence = 0.2, double pixelConfedence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_detection).ProcessImage(img, confidence, pixelConfedence, iou);

        #endregion

        #region Pose Estimation

        /// <summary>
        /// Run pose estimation on an image.
        /// </summary>
        /// <param name="img">The SKBitmap to pose estimate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<PoseEstimation> RunPoseEstimation(SKBitmap img, double confidence = 0.2, double iou = 0.7)
            => ((IPoseEstimationModule)_detection).ProcessImage(img, confidence, 0, iou);

        /// <summary>
        /// Run pose estimation on an image.
        /// </summary>
        /// <param name="img">The SKImage to pose estimate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.2).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<PoseEstimation> RunPoseEstimation(SKImage img, double confidence = 0.2, double iou = 0.7)
            => ((IPoseEstimationModule)_detection).ProcessImage(img, confidence, 0, iou);

        #endregion

        #region Video

        /// <summary>
        /// Initializes the video stream using the specified <see cref="VideoOptions"/> and sets up event handlers for frame processing and video completion.
        /// </summary>
        /// <param name="videoOptions"></param>
        public void InitializeVideo(VideoOptions videoOptions)
        {
            if (options.Cuda is false)
                throw new ArgumentException(
                    "Video inference requires CUDA acceleration (GPU support) and FFmpeg installed on your system. " +
                    "Please enable CUDA by setting 'YoloOptions.Cuda = true' in your configuration, " +
                    "and ensure FFmpeg is installed and accessible in your system PATH.");

            _ffmpegService = new(videoOptions, options)
            {
                OnFrameReady = (frame, frameIndex) => OnVideoFrameReceived.Invoke(frame, frameIndex),
                OnVideoEnd = () => OnVideoEnd?.Invoke()
            };
        }

        /// <summary>
        /// Retrieves a list of available video input devices detected on the current system.
        /// </summary>
        /// <exception cref="Exception"></exception>
        public static List<string> GetVideoDevices()
            => FFmpegService.GetVideoDevicesOnSystem() ?? throw new Exception(
                "No video initialized. Please call InitializeVideo() before attempting to retrieve metadata.");

        /// <summary>
        /// Retrieves metadata about the stream or initialized video, such as duration, frame rate, and resolution.
        /// </summary>
        /// <exception cref="Exception"></exception>
        public VideoMetadata GetVideoMetaData()
            => _ffmpegService.VideoMetadata ?? throw new Exception(
                "No video initialized. Please call InitializeVideo() before attempting to retrieve metadata.");

        /// <summary>
        /// Starts decoding and processing video frames from the initialized video stream.
        /// </summary>
        public void StartVideoProcessing()
            => _ffmpegService.Start();

        /// <summary>
        /// Stops video frame processing and releases resources associated with the video stream.
        /// </summary>
        public void StopVideoProcessing()
            => _ffmpegService.Stop();

        #endregion

        #region Dispose

        public void Dispose()
        {
            _detection.Dispose();
            _ffmpegService?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}