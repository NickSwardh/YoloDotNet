namespace YoloDotNet.Modules.V8
{
    internal class OBBDetectionModuleV8 : IOBBDetectionModule
    {
        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public OBBDetectionModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);
        }

        public List<OBBDetection> ProcessImage(SKBitmap image, double confidence, double pixelConfidence, double iou)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(image);
            var ortSpan = ortValues[0].GetTensorDataAsSpan<float>();

            var objectDetectionResults = _objectDetectionModule.ObjectDetection(image, ortSpan, confidence, iou);

            return [.. objectDetectionResults.Select(x => (OBBDetection)x)];
        }

        #region Helper methods

        public void Dispose()
        {
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
