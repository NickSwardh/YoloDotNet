// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

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

            // Yolov11 uses the YOLOv8 model architecture.
            _objectDetectionModuleV8 = new ObjectDetectionModuleV8(_yoloCore);
        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
            => _objectDetectionModuleV8.ProcessImage(image, confidence, pixelConfidence, iou, roi);

        public void Dispose()
        {
            _objectDetectionModuleV8?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

    }
}