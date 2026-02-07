// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V26
{
    internal class OBBDetectionModuleV26 : IOBBDetectionModule
    {
        private readonly YoloCore _yoloCore;
        private readonly int _totalElements;
        private readonly int _stride;
        private List<OBBDetection> _results = default!;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public OBBDetectionModuleV26(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Get output shape from ONNX model. Format: [Batch, Attributes, Predictions]
            var outputShape = _yoloCore.OnnxModel.OutputShapes.ElementAt(0).Value;
            var modelOutputChannels = outputShape[1];
            var modelOutputElements = outputShape[2];
            _totalElements = modelOutputChannels * modelOutputElements;

            _stride = modelOutputElements;
            _results = [];

            // Override image resize to Proportional for OBB detection
            // OBB requires proportional resizing to maintain geometric validity
            _yoloCore.YoloOptions.ImageResize = ImageResize.Proportional;
        }

        public List<OBBDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
        {
            var inferenceResult = _yoloCore.Run(image);
            var detections = ObjectDetection(inferenceResult, confidence, iou);

            // Convert to List<ObjectDetection>
            _results.Clear();
            for (int i = 0; i < detections.Length; i++)
                _results.Add((OBBDetection)detections[i]);

            return _results;
        }

        #region Helper methods
        [MethodImpl(MethodImplOptions.AggressiveInlining)] // Inline this small method better performance
        public Span<ObjectResult> ObjectDetection(InferenceResult inferenceResult, double confidenceThreshold, double overlapThreshold)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);

            int validBoxCount = 0;
            var boxes = ArrayPool<ObjectResult>.Shared.Rent(_totalElements);

            try
            {

                for (var i = 0; i < ortSpan.Length; i += _stride)
                {
                    // Confidence is at index 4
                    var confidence = ortSpan[i + 4];

                    // Early exit before reading other values
                    if (confidence < confidenceThreshold)
                        continue;

                    var x = ortSpan[i];
                    var y = ortSpan[i + 1];
                    var w = ortSpan[i + 2];
                    var h = ortSpan[i + 3];
                    var labelIndex = ortSpan[i + 5];
                    var angle = ortSpan[i + 6];

                    int xMin, yMin, xMax, yMax;

                    var halfW = w / 2;
                    var halfH = h / 2;

                    // IMPORTANT:
                    // Preprocessing must use proportional (uniform) resizing.
                    // OBB output is only geometrically valid under proportional (uniform) resizing.
                    // Stretched resizing applies non-uniform scaling, turning rotated rectangles
                    // into sheared shapes and making (x, y, w, h, angle) representations invalid.

                    // The model outputs are center-x/center-y/width/height.
                    // Convert to corner coordinates using half width/height and apply padding+gain.
                    xMin = (int)((x - halfW - xPad) * xGain);
                    yMin = (int)((y - halfH - yPad) * xGain);
                    xMax = (int)((x + halfW - xPad) * xGain);
                    yMax = (int)((y + halfH - yPad) * xGain);

                    var boundingBox = new SKRectI(xMin, yMin, xMax, yMax);
                    var boundingBoxUnscaled = new SKRectI((int)x, (int)y, (int)w, (int)h);

                    boxes[validBoxCount++] = new ObjectResult
                    {
                        Label = _yoloCore.OnnxModel.Labels[(int)labelIndex],
                        Confidence = confidence,
                        BoundingBox = boundingBox,
                        BoundingBoxUnscaled = boundingBoxUnscaled,
                        BoundingBoxIndex = i,
                        OrientationAngle = angle
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
