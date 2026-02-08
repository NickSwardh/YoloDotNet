// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V11
{
    internal class OBBDetectionModuleV11 : IOBBDetectionModule
    {
        private readonly YoloCore _yoloCore;
        private readonly OBBDetectionModuleV8 _obbDetectionModuleV8 = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public OBBDetectionModuleV11(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov11 uses the YOLOv8 model architecture.
            _obbDetectionModuleV8 = new OBBDetectionModuleV8(_yoloCore);
        }

        public List<OBBDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
            => _obbDetectionModuleV8.ProcessImage(image, confidence, pixelConfidence, iou, roi);

        #region Helper methods

        public void Dispose()
        {
            _obbDetectionModuleV8?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
