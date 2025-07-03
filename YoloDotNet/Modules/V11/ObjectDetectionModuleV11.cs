namespace YoloDotNet.Modules.V11
{
    internal class ObjectDetectionModuleV11 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;
        
        public OnnxModel OnnxModel => _yoloCore.OnnxModel;
        private readonly ObjectDetectionModuleV8 _objectDetectionModuleV8 = default!;

        public ObjectDetectionModuleV11(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov11 has the same model input/output shapes as Yolov8
            // Use Yolov8 module
            _objectDetectionModuleV8 = new ObjectDetectionModuleV8(_yoloCore);
        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
            => _objectDetectionModuleV8.ProcessImage(image, confidence, pixelConfidence, iou);

        public void Dispose()
        {
            _objectDetectionModuleV8?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

    }
}