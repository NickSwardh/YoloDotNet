// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V8
{
    internal class SegmentationModuleV8 : ISegmentationModule
    {
        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule;
        private readonly int _inputWidth;
        private readonly int _inputHeight;
        private readonly int _maskWidth;
        private readonly int _maskHeight;
        private readonly int _elements;
        private readonly int _predictions;
        private readonly int _outputShapeMaskChannels;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        // Represents a fixed-size float buffer of 32 elements for mask weights.
        // Uses the InlineArray attribute to avoid heap allocations entirely.
        // This structure is stack-allocated when used inside methods or structs,
        // making it ideal for high-performance scenarios where per-frame allocations must be avoided.
        [InlineArray(32)]
        internal struct MaskWeights32
        {
            private float _mask;
        }

        public SegmentationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);

            // Get input shape from ONNX model. Format NCHW: [Batch (B), Channels (C), Height (H), Width (W)]
            var inputShape = _yoloCore.OnnxModel.InputShapes.ElementAt(0).Value;

            // Get output shape from ONNX model. Format: [Batch, Attributes, Predictions]
            var outputShape = _yoloCore.OnnxModel.OutputShapes.ElementAt(0).Value;

            // Get output shape from ONNX model. Format: [Batch (B), Channels (C), Height (H), Width (W)]
            var outputShapeMask = _yoloCore.OnnxModel.OutputShapes.ElementAt(1).Value;

            // Get model pixel mask widh and height
            _maskHeight = outputShapeMask[2];
            _maskWidth = outputShapeMask[3];

            _elements = _yoloCore.OnnxModel.Labels.Length + 4; // 4 = the boundingbox dimension (x, y, width, height)
            _predictions = outputShape[2];
            _outputShapeMaskChannels = outputShapeMask[1];

            // Get model input width and height
            _inputHeight = (int)inputShape[2];
            _inputWidth = (int)inputShape[3];
        }

        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            using var inferenceResult = _yoloCore.Run(image, roi);
            var detections = RunSegmentation(inferenceResult, confidence, pixelConfidence, iou);

            return YoloCore.InferenceResultsToType(detections, roi, r => (Segmentation)r);
        }

        private Span<ObjectResult> RunSegmentation(InferenceResult inferenceResult, double confidence, double pixelConfidence, double iou)
        {
            var boundingBoxes = _objectDetectionModule.ObjectDetection(inferenceResult, confidence, iou);

            var ortSpan0 = inferenceResult.OrtSpan0;
            var ortSpan1 = inferenceResult.OrtSpan1;

            var maskTransform = MaskTransform.Create(
                _yoloCore.YoloOptions.ImageResize,
                _yoloCore.CalculateGain(inferenceResult.ImageOriginalSize),
                _inputWidth, _inputHeight, _maskWidth, _maskHeight);

            foreach (var box in boundingBoxes)
            {
                // 1) Get weights
                var maskWeights = GetMaskWeightsFromBoundingBoxArea(box, ortSpan0);

                // 2) Trace the mask outlines for the bounding box (original image coords); the pixel mask
                // is rasterized from them when needed.
                box.Contours = SegmentationMaskProcessor.CreateContours(
                    ortSpan1,
                    maskWeights,
                    _outputShapeMaskChannels,
                    _maskWidth,
                    _maskHeight,
                    box.BoundingBox,
                    maskTransform,
                    pixelConfidence);
            }

            return boundingBoxes;
        }

        private MaskWeights32 GetMaskWeightsFromBoundingBoxArea(ObjectResult box, ReadOnlySpan<float> ortSpan0)
        {
            MaskWeights32 maskWeights = default;

            var maskOffset = box.BoundingBoxIndex + (_predictions * _elements);

            for (var m = 0; m < _outputShapeMaskChannels; m++, maskOffset += _predictions)
                maskWeights[m] = ortSpan0[maskOffset];

            return maskWeights;
        }

        public void Dispose()
        {
            _objectDetectionModule?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}
