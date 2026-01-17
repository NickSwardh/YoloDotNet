// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V12
{
    internal class SegmentationModuleV12 : ISegmentationModule
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

        private readonly YoloCore _yoloCore;
        private readonly SegmentationModuleV8 _segmentationModuleV8 = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public SegmentationModuleV12(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolov12 uses the YOLOv8 model architecture.
            _segmentationModuleV8 = new SegmentationModuleV8(_yoloCore);
        }

        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
            => _segmentationModuleV8.ProcessImage(image, confidence, pixelConfidence, iou);

        #region Helper methods

        public void Dispose()
        {
            _segmentationModuleV8?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion

    }
}
