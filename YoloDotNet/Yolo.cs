namespace YoloDotNet
{
    public class Yolo : IDisposable
    {
        #region Fields

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        private readonly IDetectionModule _detection;

        public OnnxModel OnnxModel => _detection.OnnxModel;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the Yolo object detection model, which detects objects in an image or video based on a Yolo model in ONNX format.
        /// </summary>
        /// <param name="options">Options for initializing the YoloDotNet model.</param>
        public Yolo(YoloOptions options)
        {
            _detection = ModuleFactory.CreateModule(options);

            // Forward events from the detection module to the facade class
            _detection.VideoProgressEvent += (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _detection.VideoCompleteEvent += (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _detection.VideoStatusEvent += (sender, e) => VideoStatusEvent?.Invoke(sender, e);
        }

        #endregion

        #region Exposed methods for running inference on images

        /// <summary>
        /// Run image classification on an Image.
        /// </summary>
        /// <param name="img">The image to classify.</param>
        /// <param name="classes">The number of classes to return (default is 1).</param>
        /// <returns>A list of classification results.</returns>
        public List<Classification> RunClassification(SKImage img, int classes = 1)
            => ((ClassificationModule)_detection).ProcessImage(img, classes, 0);

        /// <summary>
        /// Run object detection on an Image.
        /// </summary>
        /// <param name="img">The image to classify.</param>
        /// <param name="classes">The number of classes to return (default is 1).</param>
        /// <returns>A list of classification results.</returns>
        public List<ObjectDetection> RunObjectDetection(SKImage img, double confidence = 0.25, double iou = 0.45)
            => ((ObjectDetectionModule)_detection).ProcessImage(img, confidence, iou);

        /// <summary>
        /// Run oriented bounding bBox detection on an image.
        /// </summary>
        /// <param name="img">The image to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<OBBDetection> RunObbDetection(SKImage img, double confidence = 0.25, double iou = 0.45)
            => ((ObbDetectionModule)_detection).ProcessImage(img, confidence, iou);

        /// <summary>
        /// Run segmentation on an image.
        /// </summary>
        /// <param name="img">The image to segmentate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<Segmentation> RunSegmentation(SKImage img, double confidence = 0.25, double iou = 0.45)
            => ((SegmentationModule)_detection).ProcessImage(img, confidence, iou);

        /// <summary>
        /// Run pose estimation on an image.
        /// </summary>
        /// <param name="img">The image to pose estimate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<PoseEstimation> RunPoseEstimation(SKImage img, double confidence = 0.25, double iou = 0.45)
            => ((PoseEstimationModule)_detection).ProcessImage(img, confidence, iou);

        #endregion

        #region Exposed methods for running inference on video

        /// <summary>
        /// Run image classification on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="classes">The number of classes to return for each frame (default is 1).</param>
        public Dictionary<int, List<Classification>> RunClassification(VideoOptions options, int classes = 1)
            => ((ClassificationModule)_detection).ProcessVideo(options, classes, 0);

        /// <summary>
        /// Run object detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        public Dictionary<int, List<ObjectDetection>> RunObjectDetection(VideoOptions options, double confidence = 0.25, double iou = 0.45)
            => ((ObjectDetectionModule)_detection).ProcessVideo(options, confidence, iou);

        /// <summary>
        /// Run oriented bounding box detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        public Dictionary<int, List<OBBDetection>> RunObbDetection(VideoOptions options, double confidence = 0.25, double iou = 0.45)
            => ((ObbDetectionModule)_detection).ProcessVideo(options, confidence, iou);

        /// <summary>
        /// Run object detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        public Dictionary<int, List<Segmentation>> RunSegmentation(VideoOptions options, double confidence = 0.25, double iou = 0.45)
            => ((SegmentationModule)_detection).ProcessVideo(options, confidence, iou);

        /// <summary>
        /// Run pose estimation on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        public Dictionary<int, List<PoseEstimation>> RunPoseEstimation(VideoOptions options, double confidence = 0.25, double iou = 0.45)
            => ((PoseEstimationModule)_detection).ProcessVideo(options, confidence, iou);

        public void Dispose()
        {
            _detection?.Dispose();
            VideoProgressEvent -= (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            VideoCompleteEvent -= (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            VideoStatusEvent -= (sender, e) => VideoStatusEvent?.Invoke(sender, e);
            GC.SuppressFinalize(this);
        }

        #endregion
    }
}