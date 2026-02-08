// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V26
{
    internal class PoseEstimationModuleV26 : IPoseEstimationModule
    {
        private readonly YoloCore _yoloCore;
        private int _totalElements;
        private int _dimensions;
        private int _keypointDimension;
        private int _totalKeyPoints;
        private int _stride;
        private List<PoseEstimation> _results = default!;
        private KeyPoint[] _keyPoints = default!;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public PoseEstimationModuleV26(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Get output shape from ONNX model. Format: [Batch, Attributes, Predictions]
            var outputShape = _yoloCore.OnnxModel.OutputShapes.ElementAt(0).Value;
            var modelOutputChannels = outputShape[1];
            var modelOutputElements = outputShape[2];

            _stride = modelOutputElements;
            _totalElements = modelOutputChannels * modelOutputElements;
            _dimensions = 6; // Subtract dimensions (x, y, w, h, confidence, labelIndex)
            _keypointDimension = 3; // Each keypoint has x, y, confidence

            // Calculate total keypoints
            var keyPointsSize = modelOutputElements - _dimensions;
            _totalKeyPoints = keyPointsSize / _keypointDimension;

            // Initialize reusable arrays
            _results = [];
            _keyPoints = new KeyPoint[_totalKeyPoints];
        }

        public List<PoseEstimation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            var inferenceResult = _yoloCore.Run(image, roi);
            var detections = ObjectDetection(inferenceResult, confidence, iou);

            return YoloCore.InferenceResultsToType(detections, roi, _results, r => (PoseEstimation)r);
        }

        #region Helper methods
        [MethodImpl(MethodImplOptions.AggressiveInlining)] // Inline this small method better performance
        private Span<ObjectResult> ObjectDetection(InferenceResult inferenceResult, double confidenceThreshold, double overlapThreshold)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);

            int validBoxCount = 0;
            var boxes = ArrayPool<ObjectResult>.Shared.Rent(_totalElements);

            var keypointDataSize = _totalKeyPoints * _keypointDimension;

            try
            {
                for (var i = 0; i < ortSpan.Length; i += _stride)
                {
                    var confidence = ortSpan[i + 4];
                    if (confidence < confidenceThreshold) continue;

                    var x = ortSpan[i];
                    var y = ortSpan[i + 1];
                    var w = ortSpan[i + 2];
                    var h = ortSpan[i + 3];
                    var labelIndex = ortSpan[i + 5];

                    int xMin, yMin, xMax, yMax, keyPointIndex = 0;
                    var offset = i + _dimensions;

                    if (_yoloCore.YoloOptions.ImageResize == ImageResize.Proportional)
                    {
                        xMin = (int)((x - xPad) * xGain);
                        yMin = (int)((y - yPad) * xGain);
                        xMax = (int)((w - xPad) * xGain);
                        yMax = (int)((h - yPad) * xGain);

                        // Extract keypoints
                        for (int k = offset; k < keypointDataSize + offset; k += 3)
                        {
                            // Get keypoint coordinates and confidence and rescale to original image size
                            var keypointX = (int)((ortSpan[k] - xPad) * xGain);
                            var keypointY = (int)((ortSpan[k + 1] - yPad) * xGain);
                            var keypointConf = ortSpan[k + 2];

                            _keyPoints[keyPointIndex++] = new KeyPoint(keypointX, keypointY, keypointConf);
                        }
                    }
                    else // Stretched
                    {
                        xMin = (int)(x / xGain);
                        yMin = (int)(y / yGain);
                        xMax = (int)(w / xGain);
                        yMax = (int)(h / yGain);

                        // Extract keypoints
                        for (int k = offset; k < keypointDataSize + offset; k += 3)
                        {
                            // Get keypoint coordinates and confidence and rescale to original image size
                            var keypointX = (int)(ortSpan[k] / xGain);
                            var keypointY = (int)(ortSpan[k + 1] / yGain);
                            var keypointConf = ortSpan[k + 2];

                            _keyPoints[keyPointIndex++] = new KeyPoint(keypointX, keypointY, keypointConf);
                        }
                    }

                    var boundingBox = new SKRectI(xMin, yMin, xMax, yMax);
                    var boundingBoxUnscaled = new SKRectI((int)x, (int)y, (int)w, (int)h);

                    boxes[validBoxCount++] = new ObjectResult
                    {
                        Label = _yoloCore.OnnxModel.Labels[(int)labelIndex],
                        Confidence = confidence,
                        BoundingBox = boundingBox,
                        BoundingBoxUnscaled = boundingBoxUnscaled,
                        BoundingBoxIndex = i,
                        KeyPoints = [.. _keyPoints]
                    };
                }

                return boxes.AsSpan(0, validBoxCount);
            }
            finally
            {
                ArrayPool<ObjectResult>.Shared.Return(boxes);
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