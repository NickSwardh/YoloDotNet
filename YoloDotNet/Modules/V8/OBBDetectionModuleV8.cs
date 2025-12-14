// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

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

        public List<OBBDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
        {
            var inferenceResult = _yoloCore.Run(image);

            var detections = _objectDetectionModule.ObjectDetection(inferenceResult, confidence, iou);

            // Convert to List<OBBDetection>
            var results = new List<OBBDetection>(detections.Length);
            for (int i = 0; i < detections.Length; i++)
                results.Add((OBBDetection)detections[i]);

            return results;
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
