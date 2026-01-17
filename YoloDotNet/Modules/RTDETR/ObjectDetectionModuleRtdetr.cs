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
        private List<ObjectDetection> _results = default!;
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

            _results = [];
        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
        {
            var inferenceResult = _yoloCore.Run(image);
            var detections = ObjectDetection(inferenceResult, confidence, iou);

            // Convert to List<ObjectDetection>
            _results.Clear();
            for (int i = 0; i < detections.Length; i++)
                _results.Add((ObjectDetection)detections[i]);

            return _results;
        }

        #region Helper methods
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
                        halfHeightScaled = (h * _modelInputHeight * 0.5f) * xGain;
                        xCenterOriginal = (x * _modelInputWidth - xPad) * xGain;
                        yCenterOriginal = (y * _modelInputHeight - yPad) * xGain;
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
                    int xMinTemp = (int)(xCenterOriginal - halfWidthScaled);
                    int yMinTemp = (int)(yCenterOriginal - halfHeightScaled);
                    int xMaxTemp = (int)(xCenterOriginal + halfWidthScaled);
                    int yMaxTemp = (int)(yCenterOriginal + halfHeightScaled);

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
                return boxes.AsSpan(0, validBoxCount);
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