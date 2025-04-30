namespace YoloDotNet
{
        /// <summary>
        /// Initializes a new instance of the Yolo object detection model, which detects objects in an image or video based on a Yolo model in ONNX format.
        /// </summary>
        /// <param name="options">Options for initializing the YoloDotNet model.</param>
    public class Yolo(YoloOptions options) : IDisposable
        {
        #region Fields

        private readonly IModule _detection = ModuleFactory.CreateModule(options);
        private FFMPEGService _ffmpegService = default!;
        public Action<SKBitmap, long> OnVideoFrameReceived = default!;
        public OnnxModel OnnxModel => _detection.OnnxModel;

        #endregion

        #region Exposed methods for running inference on images
    
        /// <summary>
        /// Run image classification on an Image.
        /// </summary>
        /// <param name="img">The image to classify.</param>
        /// <param name="classes">The number of classes to return (default is 1).</param>
        /// <returns>A list of classification results.</returns>
        public List<Classification> RunClassification(SKImage img, int classes = 1)
            => ((IClassificationModule)_detection).ProcessImage(img, classes, 0, 0);

        /// <summary>
        /// Run object detection on an Image.
        /// </summary>
        /// <param name="img">The image to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of classification results.</returns>
        public List<ObjectDetection> RunObjectDetection(SKImage img, double confidence = 0.23, double iou = 0.7)
            => ((IObjectDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);
        
        /// <summary>
        /// Run oriented bounding bBox detection on an image.
        /// </summary>
        /// <param name="img">The image to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<OBBDetection> RunObbDetection(SKImage img, double confidence = 0.23, double iou = 0.7)
            => ((IOBBDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);

        /// <summary>
        /// Run segmentation on an image.
        /// </summary>
        /// <param name="img">The image to segmentate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<Segmentation> RunSegmentation(SKImage img, double confidence = 0.23, double pixelConfedence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_detection).ProcessImage(img, confidence, pixelConfedence, iou);
        
        /// <summary>
        /// Run pose estimation on an image.
        /// </summary>
        /// <param name="img">The image to pose estimate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<PoseEstimation> RunPoseEstimation(SKImage img, double confidence = 0.23, double iou = 0.7)
            => ((IPoseEstimationModule)_detection).ProcessImage(img, confidence, 0, iou);

        #endregion

        #region Exposed methods for running inference on video

        public void InitializeVideo(VideoOptions options)
        {
            _ffmpegService = new(options);
            _ffmpegService.OnFrameReady = (frame, frameIndex) => OnVideoFrameReceived.Invoke(frame, frameIndex);
        }

        public VideoMetadata GetVideoMetaData()
            => _ffmpegService.VideoMetadata ?? throw new Exception(
                "No video initialized. Please call InitializeVideo() before attempting to retrieve metadata.");
        
        public void StartVideoProcessing()
            => _ffmpegService.Start();

        public void StopVideoProcessing()
            => _ffmpegService.Stop();

        public void Dispose()
        {
            _detection.Dispose();
            _ffmpegService?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}