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

        public List<ObjectDetection> ProcessImage(SKBitmap image, double confidence, double pixelConfidence, double iou)
        {
            using var ortValues = _yoloCore!.Run(image);
            using var ort = ortValues[0];
            var results = ObjectDetection(image, ort, confidence, iou)
                .Select(x => (ObjectDetection)x);

            return [.. results];
        }

        #region Helper methods

        private ObjectResult[] ObjectDetection(SKBitmap image, OrtValue ortTensor, double confidenceThreshold, double overlapThreshold)
        {
            // TODO: Implement for stretched input images too.
            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(image);

            var channels = _yoloCore.OnnxModel.Outputs[0].Channels;
            var elements = _yoloCore.OnnxModel.Outputs[0].Elements;

            var boxes = _yoloCore.customSizeObjectResultPool.Rent(channels * elements);

            var width = image.Width;
            var height = image.Height;

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

                    boxes[i] = new ObjectResult
                    {
                        Label = _yoloCore.OnnxModel.Labels[(int)labelIndex],
                        Confidence = confidence,
                        BoundingBox = new SKRectI(xMin, yMin, xMax, yMax),
                        BoundingBoxIndex = i
                    };
                }

                var results = boxes.Where(x => x is not null);
                return _yoloCore.RemoveOverlappingBoxes([.. results], overlapThreshold);
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