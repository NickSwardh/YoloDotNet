namespace YoloDotNet.Modules.V8
{
    public class ObjectDetectionModuleV8 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObjectDetectionModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            SubscribeToVideoEvents();
        }

        public List<ObjectDetection> ProcessImage(SKImage image, double confidence, double iou)
        {
            using var ortValues = _yoloCore.Run(image);
            using var ort = ortValues[0];
            var results = ObjectDetection(image, ort, confidence, iou)
                .Select(x => (ObjectDetection)x)
                .ToList();

            return results;
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
        public ObjectResult[] ObjectDetection(SKImage image, OrtValue ortTensor, double confidenceThreshold, double overlapThreshold)
        {
            var (xPad, yPad, gain) = _yoloCore.CalculateGain(image);

            var labels = _yoloCore.OnnxModel.Labels.Length;
            var channels = _yoloCore.OnnxModel.Outputs[0].Channels;

            var boxes = _yoloCore.customSizeObjectResultPool.Rent(channels);

            try
            {
                // Get tensor as a flattened Span for faster processing.
                var ortSpan = ortTensor.GetTensorDataAsSpan<float>();

                for (int i = 0; i < channels; i++)
                {
                    // Move forward to confidence value of first label
                    var labelOffset = i + channels * 4;

                    // Iterate through all labels...
                    for (var l = 0; l < labels; l++, labelOffset += channels)
                    {
                        var boxConfidence = ortSpan[labelOffset];

                        if (boxConfidence < confidenceThreshold) continue;

                        float x = ortSpan[i];                   // x
                        float y = ortSpan[i + channels];        // y
                        float w = ortSpan[i + channels * 2];    // w
                        float h = ortSpan[i + channels * 3];    // h

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

                        if (boxes[i] is null || boxConfidence > boxes[i].Confidence)
                        {
                            boxes[i] = new ObjectResult
                            {
                                Label = _yoloCore.OnnxModel.Labels[l],
                                Confidence = boxConfidence,
                                BoundingBox = new SKRectI(xMin, yMin, xMax, yMax),
                                BoundingBoxOrg = new SKRectI(sxMin, syMin, sxMax, syMax),
                                BoundingBoxIndex = i,
                                OrientationAngle = _yoloCore.OnnxModel.ModelType == ModelType.ObbDetection ? ortSpan[i + channels * (4 + labels)] : 0 //CalculateRadianToDegree(ortSpan[i + channels * (4 + labels)]) : 0, // Angle (radian) for OBB is located at the end of the labels.
                            };
                        }
                    }
                }

                var results = boxes.Where(x => x is not null).ToArray();

                return _yoloCore.RemoveOverlappingBoxes(results, overlapThreshold);
            }
            finally
            {
                _yoloCore.customSizeObjectResultPool.Return(boxes, clearArray: true);
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