namespace YoloDotNet.Modules.V12
{
    public class ClassificationModuleV12 : IClassificationModule
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

        private readonly YoloCore _yoloCore;
        private readonly ClassificationModuleV8 _classificationModuleV8 = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ClassificationModuleV12(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov11 has the same model input/output shapes as Yolov8
            // Use Yolov8 module
            _classificationModuleV8 = new ClassificationModuleV8(_yoloCore);

            SubscribeToVideoEvents();
        }

        public Classification[] ProcessImage(SKImage image, double classes, double pixelConfidence,double iou)
            => _classificationModuleV8.ProcessImage(image, classes, pixelConfidence, iou);

        public Dictionary<int, Classification[]> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou)
            => _yoloCore.RunVideo(options, confidence, pixelConfidence, iou, ProcessImage);

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

            _classificationModuleV8?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
