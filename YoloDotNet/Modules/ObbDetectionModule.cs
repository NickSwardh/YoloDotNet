namespace YoloDotNet.Modules
{
    internal class ObbDetectionModule
        : IDetectionModule, IModule<List<OBBDetection>, Dictionary<int, List<OBBDetection>>>
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModule _objectDetectionModule;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObbDetectionModule(string onnxModel, bool cuda = true, bool primeGpu = false, int gpuId = 0)
        {
            _yoloCore = new YoloCore(onnxModel, cuda, primeGpu, gpuId);
            _objectDetectionModule = new ObjectDetectionModule(_yoloCore);
            _yoloCore.InitializeYolo(ModelType.ObbDetection);
            SubscribeToVideoEvents();
        }

        public List<OBBDetection> ProcessImage(SKImage image, double confidence, double iou)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(image);
            var objectDetectionResults = _objectDetectionModule.ObjectDetectImage(image, ortValues[0], confidence, iou);

            return objectDetectionResults.Select(x => (OBBDetection)x).ToList();
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
            _yoloCore?.Dispose();
        }

        #endregion
    }
}
