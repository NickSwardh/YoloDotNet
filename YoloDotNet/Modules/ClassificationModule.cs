namespace YoloDotNet.Modules
{
    public class ClassificationModule
        : IDetectionModule, IModule<List<Classification>, Dictionary<int, List<Classification>>>
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

        private readonly YoloCore _yoloCore;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ClassificationModule(string onnxModel, bool cuda = true, bool primeGpu = false, int gpuId = 0)
        {
            _yoloCore = new YoloCore(onnxModel, cuda, primeGpu, gpuId);
            _yoloCore.InitializeYolo(ModelType.Classification);
            SubscribeToVideoEvents();
        }

        public List<Classification> ProcessImage(SKImage image, double classes, double iou)
        {
            using var ortValues = _yoloCore.Run(image);
            using var ort = ortValues[0];
            return ClassifyTensor(ort, (int)classes);
        }

        public Dictionary<int, List<Classification>> ProcessVideo(VideoOptions options, double confidence, double iou)
            => _yoloCore.RunVideo(options, confidence, iou, ProcessImage);

        #region Classicifation

        /// <summary>
        /// Classifies a tensor and returns a Classification list 
        /// </summary>
        /// <param name="numberOfClasses"></param>
        private List<Classification> ClassifyTensor(OrtValue ortTensor, int numberOfClasses)
        {
            var span = ortTensor.GetTensorDataAsSpan<float>();
            var tmp = new Classification[span.Length];

            for (int i = 0; i < tmp.Length; i++)
            {
                tmp[i] = new Classification
                {
                    Confidence = span[i],
                    Label = _yoloCore.OnnxModel.Labels[i].Name
                };
            }

            return tmp.OrderByDescending(x => x.Confidence)
                .Take(numberOfClasses)
                .ToList();
        }

        #endregion

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

            _yoloCore.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
