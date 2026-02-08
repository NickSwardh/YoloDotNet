// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V8
{
    internal class ObjectDetectionModuleV8 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;
        private readonly int _labels;
        private readonly int _attribute;
        private readonly int _attribute2;
        private readonly int _attribute3;
        private readonly int _attribute4;
        private List<ObjectDetection> _results = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObjectDetectionModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Get output shape from ONNX model. Format: [Batch, Attributes, Predictions]
            var outputShape = _yoloCore.OnnxModel.OutputShapes.ElementAt(0).Value;

            //_channels = _yoloCore.OnnxModel.Outputs[0].Channels;

            // Get number of 'Predictions' from output shape
            _attribute = outputShape[2];    // x coordinate
            _attribute2 = _attribute * 2;   // y coordinate
            _attribute3 = _attribute * 3;   // width
            _attribute4 = _attribute * 4;   // height

            _labels = _yoloCore.OnnxModel.Labels.Length;
            _results = [];
        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            var result = _yoloCore.Run(image, roi);
            var detections = ObjectDetection(result, confidence, iou);

            return YoloCore.InferenceResultsToType(detections, roi, _results, r => (ObjectDetection)r);
        }

        #region Helper methods

        /// <summary>
        /// Detects objects in a tensor and returns a ObjectDetection list.
        /// </summary>
        /// <param name="inferenceResult">The inference result containing model output data.</param>
        /// <param name="confidenceThreshold">The confidence threshold for accepting object detections.</param>
        /// <param name="overlapThreshold">The threshold for overlapping boxes to filter detections.</param>
        /// <returns>A list of result models representing detected objects.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] // Inline this small method better performance
        public Span<ObjectResult> ObjectDetection(InferenceResult inferenceResult, double confidenceThreshold, double overlapThreshold)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            if (ortSpan == null)
                return [];

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);

            var width = imageSize.Width;
            var height = imageSize.Height;

            int validBoxCount = 0;
            var boxes = ArrayPool<ObjectResult>.Shared.Rent(_attribute);

            try
            {
                for (int i = 0; i < _attribute; i++)
                {
                    // Move forward to confidence value of first label
                    var labelOffset = i + _attribute4;

                    float bestConfidence = 0f;
                    int bestLabelIndex = -1;

                    // Get confidence and label for current bounding box
                    for (var l = 0; l < _labels; l++, labelOffset += _attribute)
                    {
                        var boxConfidence = ortSpan[labelOffset];

                        if (boxConfidence > bestConfidence)
                        {
                            bestConfidence = boxConfidence;
                            bestLabelIndex = l;
                        }
                    }

                    // Stop early if confidence is low
                    if (bestConfidence < confidenceThreshold)
                        continue;

                    float x = ortSpan[i];
                    float y = ortSpan[i + _attribute];
                    float w = ortSpan[i + _attribute2];
                    float h = ortSpan[i + _attribute3];

                    int xMin, yMin, xMax, yMax;

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

                        // Manual clamping is faster than Math.Clamp in tight loops

                        int val = (int)((x - halfW - xPad) / xGain);
                        xMin = val < 0 ? 0 : (val > width - 1 ? width - 1 : val);

                        int valY = (int)((y - halfH - yPad) / yGain);
                        yMin = valY < 0 ? 0 : (valY > height - 1 ? height - 1 : valY);

                        int valXMax = (int)((x + halfW - xPad) / xGain);
                        xMax = valXMax < 0 ? 0 : (valXMax > width - 1 ? width - 1 : valXMax);

                        int valYMax = (int)((y + halfH - yPad) / yGain);
                        yMax = valYMax < 0 ? 0 : (valYMax > height - 1 ? height - 1 : valYMax);
                    }

                    // Unscaled coordinates for resized input image
                    var sxMin = (int)(x - w / 2);
                    var syMin = (int)(y - h / 2);
                    var sxMax = (int)(x + w / 2);
                    var syMax = (int)(y + h / 2);

                    var boundingBox = new SKRectI(xMin, yMin, xMax, yMax);
                    var boundingBoxUnscaled = new SKRectI(sxMin, syMin, sxMax, syMax);

                    boxes[validBoxCount++] = new ObjectResult
                    {
                        Label = _yoloCore.OnnxModel.Labels[bestLabelIndex],
                        Confidence = bestConfidence,
                        BoundingBox = boundingBox,
                        BoundingBoxUnscaled = boundingBoxUnscaled,
                        BoundingBoxIndex = i,
                        OrientationAngle = _yoloCore.OnnxModel.ModelType == ModelType.ObbDetection ? ortSpan[i + _attribute * (4 + _labels)] : 0
                    };
                }

                // Get only the valid portion of the rented array
                var resultArray = boxes.AsSpan(0, validBoxCount);

                // Remove overlapping boxes using Non-Maximum Suppression (NMS)
                return _yoloCore.RemoveOverlappingBoxes(resultArray, overlapThreshold);
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