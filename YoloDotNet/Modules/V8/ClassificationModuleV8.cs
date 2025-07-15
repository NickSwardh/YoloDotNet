namespace YoloDotNet.Modules.V8
{
    public class ClassificationModuleV8 : IClassificationModule
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

        private readonly ArrayPool<Classification> _classificationPool;
        private readonly YoloCore _yoloCore;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ClassificationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            _classificationPool = ArrayPool<Classification>.Create(maxArrayLength: OnnxModel.Outputs[0].Elements + 1, maxArraysPerBucket: 10);

            SubscribeToVideoEvents();
        }

        public Classification[] ProcessImage(SKImage image, double classes, double pixelConfidence, double iou)
        {
            using var ortValues = _yoloCore.Run(image);
            using var ort = ortValues[0];
            return ClassifyTensor(ort, (int)classes);
        }

        public Dictionary<int, Classification[]> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou)
            => _yoloCore.RunVideo(options, confidence, pixelConfidence, iou, ProcessImage);

        #region Classicifation

        /// <summary>
        /// Classifies a tensor and returns a Classification list 
        /// </summary>
        /// <param name="numberOfClasses"></param>
        private Classification[] ClassifyTensor(OrtValue ortTensor, int numberOfClasses)
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

            return [.. tmp.OrderByDescending(x => x.Confidence).Take(numberOfClasses)];
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
