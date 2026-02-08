// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

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

            // Yolov9 uses the YOLOv8 model architecture.
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);
        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
            => _objectDetectionModule.ProcessImage(image, confidence, pixelConfidence, iou, roi);

        public void Dispose()
        {
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

    }
}