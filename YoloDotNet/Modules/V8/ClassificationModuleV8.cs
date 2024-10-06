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
            var span = ortTensor.GetTensorMutableDataAsSpan<float>();
            var len = span.Length;

            var tmp = _classificationPool.Rent(len);

            try
            {
                for (int i = 0; i < len; i++)
                {
                    tmp[i] = new Classification
                    {
                        Confidence = span[i],
                        Label = _yoloCore.OnnxModel.Labels[i].Name
                    };
                }

                // Use Array.Sort() instead of LINQ for performance
                Array.Sort(tmp[.. len], (a, b) => b.Confidence.CompareTo(a.Confidence));
                return [.. tmp[..numberOfClasses]];
            }
            finally
            {
                _classificationPool.Return(tmp, true);
            }
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
