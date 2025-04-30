namespace YoloDotNet.Modules.V9
{
    internal class ObjectDetectionModuleV9 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule = default!;

        public ObjectDetectionModuleV9(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov9 has the same model input/output shapes as Yolov8
            // Use Yolov8 module
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);
        }

        public List<ObjectDetection> ProcessImage(SKBitmap image, double confidence, double pixelConfidence, double iou)
            => _objectDetectionModule.ProcessImage(image, confidence, pixelConfidence, iou);

        public void Dispose()
        {
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

    }
}