namespace YoloDotNet.Modules.V8
{
    internal class ObjectDetectionModuleV8 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;
        private static List<ObjectResult> _result = default!;
        private readonly int _labels;
        private readonly int _channels;
        private readonly int _channels2;
        private readonly int _channels3;
        private readonly int _channels4;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObjectDetectionModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            _result = [];

            _labels = _yoloCore.OnnxModel.Labels.Length;
            _channels = _yoloCore.OnnxModel.Outputs[0].Channels;
            _channels2 = _channels * 2;
            _channels3 = _channels * 3;
            _channels4 = _channels * 4;
        }

        public List<ObjectDetection> ProcessImage(SKImage image, double confidence, double pixelConfidence, double iou)
        {
            using var ortValues = _yoloCore.Run(image);
            var ortSpan = ortValues[0].GetTensorDataAsSpan<float>();

            var results = ObjectDetection(image, ortSpan, confidence, iou)
                .Select(x => (ObjectDetection)x);

            return [..results];
        }

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

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(image);

            var  boxes = _yoloCore.customSizeObjectResultPool.Rent(_channels);

            var width = image.Width;
            var height = image.Height;

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

                    var (xMin, yMin, xMax, yMax) = (0, 0, 0, 0);

                    // Bounding box calculations are based on how the input image was resized
                    // 'Proportional' keeps the original aspect ratio and adds padding around the image
                    // 'Stretched' scales the image to fill the dimensions, possibly distorting the aspect ratio
                    if (_yoloCore.YoloOptions.ImageResize == ImageResize.Proportional)
                    {
                        var gain = xGain; // Scale factor for proportional resizing

                        xMin = (int)((x - w / 2 - xPad) * gain);
                        yMin = (int)((y - h / 2 - yPad) * gain);
                        xMax = (int)((x + w / 2 - xPad) * gain);
                        yMax = (int)((y + h / 2 - yPad) * gain);
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

        public void Dispose()
        {
            _yoloCore?.Dispose();
            GC.SuppressFinalize(this);
        }

        #endregion
    }
}