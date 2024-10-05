namespace YoloDotNet.Modules.V11
{
    internal class OBBDetectionModuleV11 : IOBBDetectionModule
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModuleV8 _objectDetectionModuleV8 = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public OBBDetectionModuleV11(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov11 has the same model input/output shapes as Yolov8
            // Use Yolov8 module
            _objectDetectionModuleV8 = new ObjectDetectionModuleV8(_yoloCore);

            SubscribeToVideoEvents();
        }

        public List<OBBDetection> ProcessImage(SKImage image, double confidence, double iou)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(image);
            var ortSpan = ortValues[0].GetTensorDataAsSpan<float>();

            var objectDetectionResults = _objectDetectionModuleV8.ObjectDetection(image, ortSpan, confidence, iou);

            return [.. objectDetectionResults.Select(x => (OBBDetection)x)];
        }

        public Dictionary<int, List<OBBDetection>> ProcessVideo(VideoOptions options, double confidence, double iou)
            => _yoloCore.RunVideo(options, confidence, iou, ProcessImage);

        #region Helper methods

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

        #endregion
    }
}
