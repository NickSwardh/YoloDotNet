namespace YoloDotNet.Modules.V9
{
    public class ObjectDetectionModuleV9 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;
        
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };
        
        public OnnxModel OnnxModel => _yoloCore.OnnxModel;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule = default!;

        public ObjectDetectionModuleV9(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov9 has the same model input/output shapes as Yolov8
            // Use Yolov8 module
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);

            SubscribeToVideoEvents();
        }

        public List<ObjectDetection> ProcessImage(SKImage image, double confidence, double pixelConfidence, double iou)
            => _objectDetectionModule.ProcessImage(image, confidence, pixelConfidence, iou);

        public Dictionary<int, List<ObjectDetection>> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou)
            => _yoloCore.RunVideo(options, confidence, pixelConfidence, iou, ProcessImage);

        private void SubscribeToVideoEvents()
        {
            _yoloCore.VideoProgressEvent += (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _yoloCore.VideoCompleteEvent += (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _yoloCore.VideoStatusEvent += (sender, e) => VideoStatusEvent?.Invoke(sender, e);
        }

        public void Dispose()
        {
            _yoloCore.VideoProgressEvent -= (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _yoloCore.VideoCompleteEvent -= (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _yoloCore.VideoStatusEvent -= (sender, e) => VideoStatusEvent?.Invoke(sender, e);

            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

    }
}