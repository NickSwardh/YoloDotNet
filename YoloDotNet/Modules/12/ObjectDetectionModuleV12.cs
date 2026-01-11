// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V12
{
    internal class ObjectDetectionModuleV12 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;
        
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };
        
        public OnnxModel OnnxModel => _yoloCore.OnnxModel;
        private readonly ObjectDetectionModuleV8 _objectDetectionModuleV8 = default!;

        public ObjectDetectionModuleV12(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov12 uses the YOLOv8 model architecture.
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