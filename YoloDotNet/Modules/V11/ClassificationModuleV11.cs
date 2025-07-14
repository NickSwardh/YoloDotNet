// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V11
{
    internal class ClassificationModuleV11 : IClassificationModule
    {
        private readonly YoloCore _yoloCore;
        private readonly ClassificationModuleV8 _classificationModuleV8 = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ClassificationModuleV11(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov11 uses the YOLOv8 model architecture.
            _classificationModuleV8 = new ClassificationModuleV8(_yoloCore);
        }

        public List<Classification> ProcessImage<T>(T image, double classes, double pixelConfidence,double iou)
            => _classificationModuleV8.ProcessImage(image, classes, pixelConfidence, iou);

        #region Helper methods

        public void Dispose()
        {
            _classificationModuleV8?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
