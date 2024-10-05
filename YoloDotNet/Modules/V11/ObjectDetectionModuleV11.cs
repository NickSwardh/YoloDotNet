namespace YoloDotNet.Modules.V11
{
    public class ObjectDetectionModuleV11 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;
        
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };
        
        public OnnxModel OnnxModel => _yoloCore.OnnxModel;
        private readonly ObjectDetectionModuleV8 _objectDetectionModuleV8 = default!;

        public ObjectDetectionModuleV11(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov11 has the same model input/output shapes as Yolov8
            // Use Yolov8 module
            _objectDetectionModuleV8 = new ObjectDetectionModuleV8(_yoloCore);

            SubscribeToVideoEvents();
        }

        public List<ObjectDetection> ProcessImage(SKImage image, double confidence, double iou)
        {
            using var ortValues = _yoloCore.Run(image);
            var ortSpan = ortValues[0].GetTensorDataAsSpan<float>();

            var results = _objectDetectionModuleV8.ObjectDetection(image, ortSpan, confidence, iou)
                .Select(x => (ObjectDetection)x);

            return [..results];
        }

        public Dictionary<int, List<ObjectDetection>> ProcessVideo(VideoOptions options, double confidence, double iou)
            => _yoloCore.RunVideo(options, confidence, iou, ProcessImage);

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

            _objectDetectionModuleV8?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

    }
}