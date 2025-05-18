namespace YoloDotNet.Modules.WorldV2
{
    internal class ObjectDetectionModuleWorldV2 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule = default!;

        public ObjectDetectionModuleWorldV2(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolo WorldV2 has the same model input/output shapes as Yolov8
            // Use Yolov8 module
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);
        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
            => _objectDetectionModule.ProcessImage(image, confidence, pixelConfidence, iou);

        public void Dispose()
        {
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}