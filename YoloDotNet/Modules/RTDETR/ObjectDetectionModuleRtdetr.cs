// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.RTDETR
{
    internal class ObjectDetectionModuleRtdetr : IObjectDetectionModule
    {
        #region Public events and properties
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;
        #endregion

        #region Private fields
        private readonly YoloCore _yoloCore;
        private readonly int _attributes;
        private readonly int _predictions;
        private readonly int _modelInputWidth;
        private readonly int _modelInputHeight;
        private readonly int _totalLabels;
        private readonly int _classOffset;

        // End-to-end exports output [Batch, Predictions, 6] with normalized [cx, cy, w, h, confidence, classId]
        private const int END_TO_END_ATTRIBUTES = 6;
        #endregion

        public ObjectDetectionModuleRtdetr(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Get input shape from ONNX model. Format NCHW: [Batch (B), Channels (C), Height (H), Width (W)]
            var inputShape = _yoloCore.OnnxModel.InputShapes.ElementAt(0).Value;

            // Get output shape from ONNX model. Format for RT-DETR models: [Batch, Predictions, Attributes]
            var outputShape = _yoloCore.OnnxModel.OutputShapes.ElementAt(0).Value;

            _predictions = outputShape[1];
            _attributes = outputShape[2];

            _modelInputHeight = (int)inputShape[2];
            _modelInputWidth = (int)inputShape[3];
            _totalLabels = OnnxModel.Labels.Length;
            _classOffset = 4;  // Offset to skip the first 4 bounding box parameters (x, y, width, height)

            // Guard against label metadata that doesn't match the number of class scores in the output tensor
            _totalLabels = Math.Min(_totalLabels, _attributes - _classOffset);

        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            using var inferenceResult = _yoloCore.Run(image, roi);

            var detections = _attributes == END_TO_END_ATTRIBUTES
                ? EndToEndObjectDetection(inferenceResult, confidence)
                : ObjectDetection(inferenceResult, confidence, iou);

            return YoloCore.InferenceResultsToType(detections, roi, r => (ObjectDetection)r);
        }

        #region Helper methods
        private Span<ObjectResult> EndToEndObjectDetection(InferenceResult inferenceResult, double confidenceThreshold)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);
            var proportional = _yoloCore.YoloOptions.ImageResize == ImageResize.Proportional;
            var labels = _yoloCore.OnnxModel.Labels;

            var boxes = new List<ObjectResult>();

            for (int i = 0; i + END_TO_END_ATTRIBUTES <= ortSpan.Length; i += END_TO_END_ATTRIBUTES)
            {
                var confidence = ortSpan[i + 4];
                var labelIndex = (int)ortSpan[i + 5];

                if (confidence < confidenceThreshold || labelIndex < 0 || labelIndex >= labels.Length)
                    continue;

                float cx = ortSpan[i] * _modelInputWidth, cy = ortSpan[i + 1] * _modelInputHeight;
                float halfW = ortSpan[i + 2] * _modelInputWidth * 0.5f, halfH = ortSpan[i + 3] * _modelInputHeight * 0.5f;
                float x1 = cx - halfW, y1 = cy - halfH, x2 = cx + halfW, y2 = cy + halfH;

                int xMin, yMin, xMax, yMax;

                if (proportional)
                {
                    xMin = (int)MathF.Round((x1 - xPad) * xGain);
                    yMin = (int)MathF.Round((y1 - yPad) * yGain);
                    xMax = (int)MathF.Round((x2 - xPad) * xGain);
                    yMax = (int)MathF.Round((y2 - yPad) * yGain);
                }
                else
                {
                    xMin = (int)MathF.Round(x1 / xGain);
                    yMin = (int)MathF.Round(y1 / yGain);
                    xMax = (int)MathF.Round(x2 / xGain);
                    yMax = (int)MathF.Round(y2 / yGain);
                }

                boxes.Add(new ObjectResult
                {
                    Label = labels[labelIndex],
                    Confidence = confidence,
                    BoundingBox = new SKRectI(
                        Math.Clamp(xMin, 0, imageSize.Width - 1),
                        Math.Clamp(yMin, 0, imageSize.Height - 1),
                        Math.Clamp(xMax, 0, imageSize.Width - 1),
                        Math.Clamp(yMax, 0, imageSize.Height - 1)),
                    BoundingBoxIndex = i
                });
            }

            return boxes.ToArray();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)] // Inline this small method better performance
        private Span<ObjectResult> ObjectDetection(InferenceResult inferenceResult, double confidenceThreshold, double overlapThreshold)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);

            int validBoxCount = 0;

            // Rent a sufficiently large array from the pool
            var boxes = ArrayPool<ObjectResult>.Shared.Rent(_attributes * _predictions);

            // RT-DETR outputs detections in the format:
            // [x, y, w, h, class_0_score, class_1_score, ..., class_N_score]

            try
            {
                for (int i = 0; i < ortSpan.Length; i += _attributes)
                {
                    // Get bounding box x, y, width and height parameters
                    var x = ortSpan[i + 0];
                    var y = ortSpan[i + 1];
                    var w = ortSpan[i + 2];
                    var h = ortSpan[i + 3];

                    // Find best class & confidence
                    var labelIndex = -1;
                    var confidence = 0f;

                    for (int c = 0; c < _totalLabels; c++)
                    {
                        float score = ortSpan[i + _classOffset + c];

                        if (score > confidence)
                        {
                            confidence = score;
                            labelIndex = c;
                        }
                    }

                    // Filter out low confidence detections and invalid class labels
                    if (labelIndex == -1 || confidence < confidenceThreshold)
                        continue;

                    // Scale bounding box to original image size
                    float halfWidthScaled, halfHeightScaled, xCenterOriginal, yCenterOriginal;

                    if (_yoloCore.YoloOptions.ImageResize == ImageResize.Proportional)
                    {
                        // Proportional: uniform scaling with letterbox padding
                        halfWidthScaled = (w * _modelInputWidth * 0.5f) * xGain;
                        halfHeightScaled = (h * _modelInputHeight * 0.5f) * yGain;
                        xCenterOriginal = (x * _modelInputWidth - xPad) * xGain;
                        yCenterOriginal = (y * _modelInputHeight - yPad) * yGain;
                    }
                    else
                    {
                        // Stretched: non-uniform scaling, no padding
                        halfWidthScaled = (w * _modelInputWidth * 0.5f) * xGain;
                        halfHeightScaled = (h * _modelInputHeight * 0.5f) * yGain;
                        xCenterOriginal = x * _modelInputWidth * xGain;
                        yCenterOriginal = y * _modelInputHeight * yGain;
                    }

                    // Calculate corners with manual clamping for better performance
                    int xMinTemp = (int)MathF.Round(xCenterOriginal - halfWidthScaled);
                    int yMinTemp = (int)MathF.Round(yCenterOriginal - halfHeightScaled);
                    int xMaxTemp = (int)MathF.Round(xCenterOriginal + halfWidthScaled);
                    int yMaxTemp = (int)MathF.Round(yCenterOriginal + halfHeightScaled);

                    // Manual clamping is faster than Math.Clamp in tight loops
                    var xMin = xMinTemp < 0 ? 0 : (xMinTemp >= imageSize.Width ? imageSize.Width - 1 : xMinTemp);
                    var yMin = yMinTemp < 0 ? 0 : (yMinTemp >= imageSize.Height ? imageSize.Height - 1 : yMinTemp);
                    var xMax = xMaxTemp < 0 ? 0 : (xMaxTemp >= imageSize.Width ? imageSize.Width - 1 : xMaxTemp);
                    var yMax = yMaxTemp < 0 ? 0 : (yMaxTemp >= imageSize.Height ? imageSize.Height - 1 : yMaxTemp);

                    boxes[validBoxCount++] = new ObjectResult
                    {
                        Label = _yoloCore.OnnxModel.Labels[(int)labelIndex],
                        Confidence = confidence,
                        BoundingBox = new SKRectI(xMin, yMin, xMax, yMax),
                        BoundingBoxIndex = i
                    };
                }

                // Get only the valid portion of the rented array
                // Copy the results out: 'boxes' goes back to the shared ArrayPool in the finally block below,
                // where any other inference call could rent and overwrite it.
                return boxes.AsSpan(0, validBoxCount).ToArray();
            }
            finally
            {
                // Return the rented array to the pool
                ArrayPool<ObjectResult>.Shared.Return(boxes, false);
            }
        }

        public void Dispose()
        {
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
