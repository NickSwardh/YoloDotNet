// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V8
{
    internal class OBBDetectionModuleV8 : IOBBDetectionModule
    {
        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule = default!;
        private List<OBBDetection> _results = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public OBBDetectionModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);
            _results = [];

            // Override image resize to Proportional for OBB detection
            // OBB requires proportional resizing to maintain geometric validity
            yoloCore.YoloOptions.ImageResize = ImageResize.Proportional;
        }

        public List<OBBDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            var inferenceResult = _yoloCore.Run(image, roi);
            var detections = _objectDetectionModule.ObjectDetection(inferenceResult, confidence, iou);

            return YoloCore.InferenceResultsToType(detections, roi, _results, r => (OBBDetection)r);
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
