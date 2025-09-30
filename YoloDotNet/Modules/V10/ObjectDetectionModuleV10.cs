// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V10
{
    internal class ObjectDetectionModuleV10 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObjectDetectionModuleV10(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
        {
            var inferenceResult = _yoloCore.Run(image);
            var detections = ObjectDetection(inferenceResult, confidence, iou);

            // Convert to List<ObjectDetection>
            var results = new List<ObjectDetection>(detections.Length);
            for (int i = 0; i < detections.Length; i++)
                results.Add((ObjectDetection)detections[i]);

            return results;
        }

        #region Helper methods

        private ObjectResult[] ObjectDetection(InferenceResult inferenceResult, double confidenceThreshold, double overlapThreshold)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);

            var channels = _yoloCore.OnnxModel.Outputs[0].Channels;
            var elements = _yoloCore.OnnxModel.Outputs[0].Elements;

            int validBoxCount = 0;
            var boxes = _yoloCore.customSizeObjectResultPool.Rent(channels * elements);

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

                    var (xMin, yMin, xMax, yMax) = (0, 0, 0, 0);

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

                // Get only the valid portion of the rented array
                var resultArray = boxes.AsSpan(0, validBoxCount);

                // Remove overlapping boxes using Non-Maximum Suppression (NMS)
                return _yoloCore.RemoveOverlappingBoxes(resultArray, overlapThreshold);
            }
            finally
            {
                _yoloCore.customSizeObjectResultPool.Return(boxes, false);
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