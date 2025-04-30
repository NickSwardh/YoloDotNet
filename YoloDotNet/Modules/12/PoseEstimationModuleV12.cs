namespace YoloDotNet.Modules.V12
{
    internal class PoseEstimationModuleV12 : IPoseEstimationModule
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

        private readonly YoloCore _yoloCore;
        private readonly PoseEstimationModuleV8 _poseEstimationModuleV8 = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public PoseEstimationModuleV12(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov12 has the same model input/output shapes as Yolov8
            // Use Yolov8 module
            _poseEstimationModuleV8 = new PoseEstimationModuleV8(_yoloCore);
        }

        public List<PoseEstimation> ProcessImage(SKBitmap image, double confidence, double pixelConfidence, double iou)
            => _poseEstimationModuleV8.ProcessImage(image, confidence, pixelConfidence, iou);

        #region Helper methods

        public void Dispose()
        {
            _poseEstimationModuleV8?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
