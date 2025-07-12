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
            var (ortValues, imageSize) = _yoloCore.Run(image);
            using IDisposableReadOnlyCollection<OrtValue> _ = ortValues;

            using var ort = ortValues[0];
            var results = ObjectDetection(imageSize, ort, confidence, iou)
                .Select(x => (ObjectDetection)x)
                .ToList();

            return results;
        }

        #region Helper methods

        private List<ObjectResult> ObjectDetection(SKSizeI imageSize, OrtValue ortTensor, double confidenceThreshold, double overlapThreshold)
        {
            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);

            var channels = _yoloCore.OnnxModel.Outputs[0].Channels;
            var elements = _yoloCore.OnnxModel.Outputs[0].Elements;

            int validBoxCount = 0;
            var boxes = _yoloCore.customSizeObjectResultPool.Rent(channels * elements);

            var width = imageSize.Width;
            var height = imageSize.Height;

            var ortSpan = ortTensor.GetTensorDataAsSpan<float>();

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

                    boxes[validBoxCount] = new ObjectResult
                    {
                        Label = _yoloCore.OnnxModel.Labels[(int)labelIndex],
                        Confidence = confidence,
                        BoundingBox = new SKRectI(xMin, yMin, xMax, yMax),
                        BoundingBoxIndex = i
                    };
                }

                // Prevent overhead from temporary collections by copying to a fixed-size array.
                var resultArray = new ObjectResult[validBoxCount];
                boxes.AsSpan(0, validBoxCount).CopyTo(resultArray);

                return _yoloCore.RemoveOverlappingBoxes(resultArray, overlapThreshold);
            }
            finally
            {
                _yoloCore.customSizeObjectResultPool.Return(boxes, true);
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