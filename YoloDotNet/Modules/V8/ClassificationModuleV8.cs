namespace YoloDotNet.Modules.V8
{
    internal class ClassificationModuleV8 : IClassificationModule
    {
        //private readonly ArrayPool<Classification> _classificationPool;
        private readonly YoloCore _yoloCore;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ClassificationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            //_classificationPool = ArrayPool<Classification>.Create(maxArrayLength: OnnxModel.Outputs[0].Elements + 1, maxArraysPerBucket: 10);
        }

        public List<Classification> ProcessImage(SKImage image, double classes, double pixelConfidence, double iou)
        {
            using var ortValues = _yoloCore.Run(image);
            using var ort = ortValues[0];
            return ClassifyTensor(ort, (int)classes);
        }

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

            return [.. tmp.OrderByDescending(x => x.Confidence).Take(numberOfClasses)];
        }

        #endregion

        #region Helper methods

        public void Dispose()
        {
            _yoloCore.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
