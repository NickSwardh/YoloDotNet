namespace YoloDotNet.Modules.V11
{
    public class SegmentationModuleV11 : ISegmentationModule
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

        private readonly YoloCore _yoloCore;
        private readonly SegmentationModuleV8 _segmentationModuleV8 = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public SegmentationModuleV11(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov11 has the same model input/output shapes as Yolov8
            // Use Yolov8 module
            _segmentationModuleV8 = new SegmentationModuleV8(_yoloCore);

            SubscribeToVideoEvents();
        }

        public List<Segmentation> ProcessImage(SKImage image, double confidence, double pixelConfidence, double iou)
            => _segmentationModuleV8.ProcessImage(image, confidence, pixelConfidence, iou);

        public Dictionary<int, List<Segmentation>> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou)
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

            _segmentationModuleV8?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion

    }
}
