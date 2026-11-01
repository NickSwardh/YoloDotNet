// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V26
{
    internal class ObjectDetectionModuleV26 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObjectDetectionModuleV26(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)] // Inline this method better performance
        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou = 0, SKRectI? roi = null)
        {
            using var inferenceResult = _yoloCore.Run(image, roi);
            return ObjectDetection(inferenceResult, confidence, roi);
        }

        #region Helper methods
        [MethodImpl(MethodImplOptions.AggressiveInlining)] // Inline this method better performance
        private List<ObjectDetection> ObjectDetection(InferenceResult inferenceResult, double confidenceThreshold, SKRectI? roi)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);

            // Count first so the final list needs only one backing array, including for crowded scenes.
            int count = 0;
            for (int i = 4; i < ortSpan.Length; i += 6)
                if (!(ortSpan[i] < confidenceThreshold))
                    count++;

            var boxes = new List<ObjectDetection>(count);
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
                    xMin = (int)MathF.Round((x - xPad) * xGain);
                    yMin = (int)MathF.Round((y - yPad) * yGain);
                    xMax = (int)MathF.Round((w - xPad) * xGain);
                    yMax = (int)MathF.Round((h - yPad) * yGain);
                }
                else
                {
                    // Rescale boxes to original image size
                    xMin = (int)MathF.Round(x / xGain);
                    yMin = (int)MathF.Round(y / yGain);
                    xMax = (int)MathF.Round(w / xGain);
                    yMax = (int)MathF.Round(h / yGain);
                }

                var boundingBox = new SKRectI(xMin, yMin, xMax, yMax);

                // Add box to results
                boxes.Add(new ObjectDetection
                {
                    Label = _yoloCore.OnnxModel.Labels[(int)labelIndex],
                    Confidence = confidence,
                    BoundingBox = roi.HasValue ? YoloCore.TransformRoiBoundingBox(boundingBox, roi.Value) : boundingBox
                });
            }

            return boxes;
        }

        public void Dispose()
        {
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
