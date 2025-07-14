// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V12
{
    internal class OBBDetectionModuleV12 : IOBBDetectionModule
    {
        private readonly YoloCore _yoloCore;
        private readonly OBBDetectionModuleV8 _obbDetectionModuleV8 = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public OBBDetectionModuleV12(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov12 uses the YOLOv8 model architecture.
            _obbDetectionModuleV8 = new OBBDetectionModuleV8(_yoloCore);
        }

        public List<OBBDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
            => _obbDetectionModuleV8.ProcessImage(image, confidence, pixelConfidence, iou);

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
