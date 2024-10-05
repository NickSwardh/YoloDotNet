namespace YoloDotNet.Modules.V8
{
    public class ObjectDetectionModuleV8 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;
        private static List<ObjectResult> _result = default!;
        private readonly int _labels;
        private readonly int _channels;
        private readonly int _channels2;
        private readonly int _channels3;
        private readonly int _channels4;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObjectDetectionModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            _result = new ();

            _labels = _yoloCore.OnnxModel.Labels.Length;
            _channels = _yoloCore.OnnxModel.Outputs[0].Channels;
            _channels2 = _channels * 2;
            _channels3 = _channels * 3;
            _channels4 = _channels * 4;

            SubscribeToVideoEvents();
        }

        public List<ObjectDetection> ProcessImage(SKImage image, double confidence, double iou)
        {
            using var ortValues = _yoloCore.Run(image);
            var ortSpan = ortValues[0].GetTensorDataAsSpan<float>();

            var results = ObjectDetection(image, ortSpan, confidence, iou)
                .Select(x => (ObjectDetection)x);

            return [..results];
        }

        public Dictionary<int, List<ObjectDetection>> ProcessVideo(VideoOptions options, double confidence, double iou)
            => _yoloCore.RunVideo(options, confidence, iou, ProcessImage);

        #region Helper methods

        /// <summary>
        /// Detects objects in a tensor and returns a ObjectDetection list.
        /// </summary>
        /// <param name="image">The image associated with the tensor data.</param>
        /// <param name="confidenceThreshold">The confidence threshold for accepting object detections.</param>
        /// <param name="overlapThreshold">The threshold for overlapping boxes to filter detections.</param>
        /// <returns>A list of result models representing detected objects.</returns>
        public ObjectResult[] ObjectDetection(SKImage image, ReadOnlySpan<float> ortSpan, double confidenceThreshold, double overlapThreshold)
        {
            if (ortSpan == null)
                return [];

            var (xPad, yPad, gain) = _yoloCore.CalculateGain(image);
  
            var  boxes = _yoloCore.customSizeObjectResultPool.Rent(_channels);
            
            try
            {
                for (int i = 0; i < _channels; i++)
                {
                    // Move forward to confidence value of first label
                    var labelOffset = i + _channels4;

                    float x = ortSpan[i];
                    float y = ortSpan[i + _channels];
                    float w = ortSpan[i + _channels2];
                    float h = ortSpan[i + _channels3]; 

                    // Scaled coordinates for original image
                    var xMin = (int)((x - w / 2 - xPad) * gain);
                    var yMin = (int)((y - h / 2 - yPad) * gain);
                    var xMax = (int)((x + w / 2 - xPad) * gain);
                    var yMax = (int)((y + h / 2 - yPad) * gain);

                    // Unscaled coordinates for resized input image
                    var sxMin = (int)(x - w / 2);
                    var syMin = (int)(y - h / 2);
                    var sxMax = (int)(x + w / 2);
                    var syMax = (int)(y + h / 2);

                    var boundingBox = new SKRectI(xMin, yMin, xMax, yMax);
                    var boundingBoxUnscaled = new SKRectI(sxMin, syMin, sxMax, syMax);

                    float bestConfidence = 0f;
                    int bestLabelIndex = -1;

                    // Iterate through all labels
                    for (var l = 0; l < _labels; l++, labelOffset += _channels)
                    {
                        var boxConfidence = ortSpan[labelOffset];

                        if (boxConfidence > bestConfidence)
                        {
                            bestConfidence = boxConfidence;
                            bestLabelIndex = l;
                        }
                    }

                    if (bestConfidence > confidenceThreshold && bestLabelIndex != -1)
                    {
                        boxes[i] = new ObjectResult
                        {
                            Label = _yoloCore.OnnxModel.Labels[bestLabelIndex],
                            Confidence = bestConfidence,
                            BoundingBox = boundingBox,
                            BoundingBoxUnscaled = boundingBoxUnscaled,
                            BoundingBoxIndex = i,
                            OrientationAngle = _yoloCore.OnnxModel.ModelType == ModelType.ObbDetection ? ortSpan[i + _channels * (4 + _labels)] : 0
                        };
                    }
                }
                
                foreach (var item in boxes)
                {
                    if (item != null)
                        _result.Add(item);
                }

                return _yoloCore.RemoveOverlappingBoxes([.. _result], overlapThreshold);
            }
            finally
            {
                _yoloCore.customSizeObjectResultPool.Return(boxes, clearArray: true);
                _result.Clear();
            }
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