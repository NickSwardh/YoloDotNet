namespace YoloDotNet
{
    public class Yolo : IDisposable
    {
        #region Fields

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        private readonly IModule _detection;

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
        public Classification[] RunClassification(SKImage img, int classes = 1)
            => ((IClassificationModule)_detection).ProcessImage(img, classes, 0, 0);

        /// <summary>
        /// Run object detection on an Image.
        /// </summary>
        /// <param name="img">The image to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of classification results.</returns>
        public ObjectDetection[] RunObjectDetection(SKImage img, double confidence = 0.23, double iou = 0.7)
            => ((IObjectDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);
        
        /// <summary>
        /// Run oriented bounding bBox detection on an image.
        /// </summary>
        /// <param name="img">The image to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public OBBDetection[] RunObbDetection(SKImage img, double confidence = 0.23, double iou = 0.7)
            => ((IOBBDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);

        /// <summary>
        /// Run segmentation on an image.
        /// </summary>
        /// <param name="img">The image to segmentate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public Segmentation[] RunSegmentation(SKImage img, double confidence = 0.23, double pixelConfedence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_detection).ProcessImage(img, confidence, pixelConfedence, iou);
        
        /// <summary>
        /// Run pose estimation on an image.
        /// </summary>
        /// <param name="img">The image to pose estimate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public PoseEstimation[] RunPoseEstimation(SKImage img, double confidence = 0.23, double iou = 0.7)
            => ((IPoseEstimationModule)_detection).ProcessImage(img, confidence, 0, iou);

        #endregion

        #region Exposed methods for running inference on video

        /// <summary>
        /// Run image classification on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="classes">The number of classes to return for each frame (default is 1).</param>
        public Dictionary<int, Classification[]> RunClassification(VideoOptions options, int classes = 1)
            => ((IClassificationModule)_detection).ProcessVideo(options, classes, 0, 0);

        /// <summary>
        /// Run object detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        public Dictionary<int, ObjectDetection[]> RunObjectDetection(VideoOptions options, double confidence = 0.23, double iou = 0.7)
            => ((IObjectDetectionModule)_detection).ProcessVideo(options, confidence, 0, iou);  //((ObjectDetectionModule)_detection).ProcessVideo(options, confidence, iou);
        
        /// <summary>
        /// Run oriented bounding box detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        public Dictionary<int, OBBDetection[]> RunObbDetection(VideoOptions options, double confidence = 0.23, double iou = 0.7)
            => ((IOBBDetectionModule)_detection).ProcessVideo(options, confidence, 0, iou);

        /// <summary>
        /// Run object detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        public Dictionary<int, Segmentation[]> RunSegmentation(VideoOptions options, double confidence = 0.23, double pixelConfidence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_detection).ProcessVideo(options, confidence, pixelConfidence, iou);

        /// <summary>
        /// Run pose estimation on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        public Dictionary<int, PoseEstimation[]> RunPoseEstimation(VideoOptions options, double confidence = 0.23, double iou = 0.7)
            => ((IPoseEstimationModule)_detection).ProcessVideo(options, confidence, 0, iou);

        public void Dispose()
        {
            VideoProgressEvent -= (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            VideoCompleteEvent -= (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            VideoStatusEvent -= (sender, e) => VideoStatusEvent?.Invoke(sender, e);

            _detection.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}