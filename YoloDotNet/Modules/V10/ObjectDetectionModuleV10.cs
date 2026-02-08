// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V10
{
    internal class ObjectDetectionModuleV10 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;
        private int _totalElements;
        private List<ObjectDetection> _results = default!;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObjectDetectionModuleV10(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Get output shape from ONNX model. Format: [Batch, Attributes, Predictions]
            var outputShape = _yoloCore.OnnxModel.OutputShapes.ElementAt(0).Value;
            var modelOutputChannels = outputShape[1];
            var modelOutputElements = outputShape[2];
            _totalElements = modelOutputChannels * modelOutputElements;

            _results = [];
        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            var inferenceResult = _yoloCore.Run(image, roi);
            var detections = ObjectDetection(inferenceResult, confidence, iou);

            return YoloCore.InferenceResultsToType(detections, roi, _results, r => (ObjectDetection)r);
        }

        #region Helper methods
        [MethodImpl(MethodImplOptions.AggressiveInlining)] // Inline this small method better performance
        private Span<ObjectResult> ObjectDetection(InferenceResult inferenceResult, double confidenceThreshold, double overlapThreshold)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);

            int validBoxCount = 0;
            //var boxes = _yoloCore.customSizeObjectResultPool.Rent(channels * elements);
            var boxes = ArrayPool<ObjectResult>.Shared.Rent(_totalElements);

            var width = imageSize.Width;
            var height = imageSize.Height;

            try
            {
                for (var i = 0; i < ortSpan.Length; i += 6)
                {
                    var x = ortSpan[i];
                    var y = ortSpan[i + 1];
                    var w = ortSpan[i + 2];
                    var h = ortSpan[i + 3];
                    var confidence = ortSpan[i + 4];
                    var labelIndex = ortSpan[i + 5];

                    if (confidence < confidenceThreshold) continue;

                    int xMin, yMin, xMax, yMax;

                    if (_yoloCore.YoloOptions.ImageResize == ImageResize.Proportional)
                    {
                        xMin = (int)((x - xPad) * xGain);
                        yMin = (int)((y - yPad) * xGain);
                        xMax = (int)((w - xPad) * xGain);
                        yMax = (int)((h - yPad) * xGain);
                    }
                    else
                    {
                        var halfW = w / 2;
                        var halfH = h / 2;

                        // Calculate bounding box coordinates adjusted for stretched scaling and padding
                        // Clamp ensures the coordinates remain within the valid bounds of the image.
                        xMin = Math.Clamp((int)((x - halfW - xPad) / xGain), 0, width - 1);
                        yMin = Math.Clamp((int)((y - halfH - yPad) / yGain), 0, height - 1);
                        xMax = Math.Clamp((int)((x + halfW - xPad) / xGain), 0, width - 1);
                        yMax = Math.Clamp((int)((y + halfH - yPad) / yGain), 0, height - 1);
                    }

                    boxes[validBoxCount++] = new ObjectResult
                    {
                        Label = _yoloCore.OnnxModel.Labels[(int)labelIndex],
                        Confidence = confidence,
                        BoundingBox = new SKRectI(xMin, yMin, xMax, yMax),
                        BoundingBoxIndex = i
                    };
                }

                return boxes.AsSpan(0, validBoxCount);
            }
            finally
            {
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