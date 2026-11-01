// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V26
{
    internal class SegmentationModuleV26 : ISegmentationModule
    {
        // Represents a fixed-size float buffer of 32 elements for mask weights.
        // Uses the InlineArray attribute to avoid heap allocations entirely.
        // This structure is stack-allocated when used inside methods or structs,
        // making it ideal for high-performance scenarios where per-frame allocations must be avoided.
        [InlineArray(32)]
        internal struct MaskWeights32
        {
            private float _mask;
        }

        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModuleV26 _objectDetectionModule;
        private readonly int _totalElements;
        private readonly int _inputWidth;
        private readonly int _inputHeight;
        private readonly int _maskWidth;
        private readonly int _maskHeight;
        private readonly int _elements;
        private readonly int _predictions;
        private readonly int _outputShapeMaskChannels;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;


        public SegmentationModuleV26(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            _objectDetectionModule = new ObjectDetectionModuleV26(_yoloCore);

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
            var inputHeight = inputShape[2];
            var inputWidth = inputShape[3];

            _totalElements = outputShape[1] * outputShape[2];

            _inputWidth = (int)inputWidth;
            _inputHeight = (int)inputHeight;
        }

        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            using var inferenceResult = _yoloCore.Run(image, roi);
            var detections = RunSegmentation(inferenceResult, confidence, pixelConfidence);

            return YoloCore.InferenceResultsToType(detections, roi, r => (Segmentation)r);
        }

        private Span<ObjectResult> RunSegmentation(InferenceResult inferenceResult, double confidenceThreshold, double pixelConfidence)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            var ortSpan1 = inferenceResult.OrtSpan1;

            var gain = _yoloCore.CalculateGain(imageSize);
            var (xPad, yPad, xGain, yGain) = gain;

            var maskTransform = MaskTransform.Create(
                _yoloCore.YoloOptions.ImageResize, gain, _inputWidth, _inputHeight, _maskWidth, _maskHeight);

            int validBoxCount = 0;
            var boxes = ArrayPool<ObjectResult>.Shared.Rent(_totalElements);

            // Per call, on the stack: a module field would be shared by concurrent calls, which would then
            // trace their masks with each other's weights.
            MaskWeights32 maskWeights = default;

            try
            {
                for (var i = 0; i < ortSpan.Length; i += _predictions)
                {
                    var confidence = ortSpan[i + 4];

                    // Early exit before reading other values
                    if (confidence < confidenceThreshold)
                        continue;

                    var x = ortSpan[i];
                    var y = ortSpan[i + 1];
                    var w = ortSpan[i + 2];
                    var h = ortSpan[i + 3];
                    var labelIndex = ortSpan[i + 5];

                    int xMin, yMin, xMax, yMax;
                    var offset = i + 6;

                    // Remaining 32 values are mask weights.
                    // Read mask weights for this bounding box
                    for (int k = 0; k < 32; k++)
                    {
                        maskWeights[k] = ortSpan[offset + k];
                    }

                    // Calculate bounding box coordinates adjusted for scaling and padding
                    if (_yoloCore.YoloOptions.ImageResize == ImageResize.Proportional)
                    {
                        xMin = (int)MathF.Round((x - xPad) * xGain);
                        yMin = (int)MathF.Round((y - yPad) * yGain);
                        xMax = (int)MathF.Round((w - xPad) * xGain);
                        yMax = (int)MathF.Round((h - yPad) * yGain);
                    }
                    // Stretched scaling
                    else
                    {
                        xMin = (int)MathF.Round(x / xGain);
                        yMin = (int)MathF.Round(y / yGain);
                        xMax = (int)MathF.Round(w / xGain);
                        yMax = (int)MathF.Round(h / yGain);
                    }

                    var boundingBox = new SKRectI(xMin, yMin, xMax, yMax);

                    boxes[validBoxCount] = new ObjectResult
                    {
                        Label = _yoloCore.OnnxModel.Labels[(int)labelIndex],
                        Confidence = confidence,
                        BoundingBox = boundingBox,
                        BoundingBoxIndex = i
                    };

                    // Trace the mask outlines for the returned BoundingBox (original image coords); the pixel
                    // mask is rasterized from them when needed.
                    boxes[validBoxCount].Contours = SegmentationMaskProcessor.CreateContours(
                        ortSpan1,
                        maskWeights,
                        _outputShapeMaskChannels,
                        _maskWidth,
                        _maskHeight,
                        boundingBox,
                        maskTransform,
                        pixelConfidence);

                    validBoxCount++;
                }

                // Copy the results out: 'boxes' goes back to the shared ArrayPool in the finally block below,
                // where any other inference call could rent and overwrite it.
                return boxes.AsSpan(0, validBoxCount).ToArray();

            }
            finally
            {
                // Return rented array
                ArrayPool<ObjectResult>.Shared.Return(boxes);
            }
        }

        public void Dispose()
        {
            _objectDetectionModule?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}
