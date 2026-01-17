// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V26
{
    internal class ObjectDetectionModuleV26 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;
        private int _totalElements;
        private List<ObjectDetection> _results = default!;
        private ObjectResult[] _boxes = default!;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObjectDetectionModuleV26(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Get output shape from ONNX model. Format: [Batch, Attributes, Predictions]
            var outputShape = _yoloCore.OnnxModel.OutputShapes.ElementAt(0).Value;
            var modelOutputChannels = outputShape[1];
            var modelOutputElements = outputShape[2];
            _totalElements = modelOutputChannels * modelOutputElements;

            _results = [];
            _boxes = new ObjectResult[_totalElements];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)] // Inline this method better performance
        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou = 0)
        {
            var inferenceResult = _yoloCore.Run(image);
            var detections = ObjectDetection(inferenceResult, confidence);

            // Convert to List<ObjectDetection>
            _results.Clear();
            for (int i = 0; i < detections.Length; i++)
                _results.Add((ObjectDetection)detections[i]);

            return _results;
        }

        #region Helper methods
        [MethodImpl(MethodImplOptions.AggressiveInlining)] // Inline this method better performance
        public Span<ObjectResult> ObjectDetection(InferenceResult inferenceResult, double confidenceThreshold)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);

            int validBoxCount = 0;

            var width = imageSize.Width;
            var height = imageSize.Height;

            for (var i = 0; i < ortSpan.Length; i += 6)
            {
                var confidence = ortSpan[i + 4];

                // Filter out low confidence boxes
                if (confidence < confidenceThreshold)
                    continue;

                // Extract box data
                var x = ortSpan[i];
                var y = ortSpan[i + 1];
                var w = ortSpan[i + 2];
                var h = ortSpan[i + 3];
                var labelIndex = ortSpan[i + 5];

                int xMin, yMin, xMax, yMax;

                if (_yoloCore.YoloOptions.ImageResize == ImageResize.Proportional)
                {
                    // Undo padding and rescale boxes to original image size
                    xMin = (int)((x - xPad) * xGain);
                    yMin = (int)((y - yPad) * xGain);
                    xMax = (int)((w - xPad) * xGain);
                    yMax = (int)((h - yPad) * xGain);
                }
                else 
                {
                    // Rescale boxes to original image size
                    xMin = (int)(x / xGain);
                    yMin = (int)(y / yGain);
                    xMax = (int)(w / xGain);
                    yMax = (int)(h / yGain);
                }

                var boundingBox = new SKRectI(xMin, yMin, xMax, yMax);
                var boundingBoxUnscaled = new SKRectI((int)x, (int)y, (int)w, (int)h);

                // Add box to results
                _boxes[validBoxCount++] = new ObjectResult
                {
                    Label = _yoloCore.OnnxModel.Labels[(int)labelIndex],
                    Confidence = confidence,
                    BoundingBox = boundingBox,
                    BoundingBoxUnscaled = boundingBoxUnscaled,
                    BoundingBoxIndex = i
                };
            }

            return _boxes.AsSpan(0, validBoxCount);
        }

        public void Dispose()
        {
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}