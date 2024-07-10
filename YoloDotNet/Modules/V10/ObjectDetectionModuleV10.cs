namespace YoloDotNet.Modules.V10
{
    public class ObjectDetectionModuleV10 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObjectDetectionModuleV10(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            SubscribeToVideoEvents();
        }

        public List<ObjectDetection> ProcessImage(SKImage image, double confidence, double iou)
        {
            using var ortValues = _yoloCore!.Run(image);
            using var ort = ortValues[0];
            var results = ObjectDetection(image, ort, confidence, iou)
                .Select(x => (ObjectDetection)x);

            return [.. results];
        }

        public Dictionary<int, List<ObjectDetection>> ProcessVideo(VideoOptions options, double confidence, double iou)
            => _yoloCore.RunVideo(options, confidence, iou, ProcessImage);

        #region Helper methods

        private ObjectResult[] ObjectDetection(SKImage image, OrtValue ortTensor, double confidenceThreshold, double overlapThreshold)
        {
            var (xPad, yPad, gain) = _yoloCore!.CalculateGain(image);

            var channels = _yoloCore.OnnxModel.Outputs[0].Channels;
            var elements = _yoloCore.OnnxModel.Outputs[0].Elements;

            var boxes = _yoloCore.customSizeObjectResultPool.Rent(channels * elements);

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

                    var minX = (int)((x - xPad) * gain);
                    var minY = (int)((y - yPad) * gain);
                    var width = (int)((w - xPad) * gain);
                    var height = (int)((h - yPad) * gain);

                    boxes[i] = new ObjectResult
                    {
                        Label = _yoloCore.OnnxModel.Labels[(int)labelIndex],
                        Confidence = confidence,
                        BoundingBox = new SKRectI((int)minX, (int)minY, (int)width, (int)height),
                        BoundingBoxIndex = i
                    };
                }
            }
            finally
            {
                _yoloCore.customSizeObjectResultPool.Return(boxes, true);
            }

            var results = boxes.Where(x => x is not null);

            return _yoloCore.RemoveOverlappingBoxes([.. results], overlapThreshold);
        }

        private void SubscribeToVideoEvents()
        {
            _yoloCore!.VideoProgressEvent += (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _yoloCore.VideoCompleteEvent += (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _yoloCore.VideoStatusEvent += (sender, e) => VideoStatusEvent?.Invoke(sender, e);
        }

        public void Dispose()
        {
            VideoProgressEvent -= (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            VideoCompleteEvent -= (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            VideoStatusEvent -= (sender, e) => VideoStatusEvent?.Invoke(sender, e);

            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}