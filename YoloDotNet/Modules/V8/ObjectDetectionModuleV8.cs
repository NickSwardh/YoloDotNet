// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using System.Numerics;

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
        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            using var result = _yoloCore.Run(image, roi);
            var detections = ObjectDetection(result, confidence, iou);

            return YoloCore.InferenceResultsToType(detections, roi, r => (ObjectDetection)r);
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
            var bestConfidences = ArrayPool<float>.Shared.Rent(_attribute);
            var bestLabelIndices = ArrayPool<int>.Shared.Rent(_attribute);

            try
            {
                // Find the best label and its confidence for every bounding box
                FindBestLabels(ortSpan, bestConfidences.AsSpan(0, _attribute), bestLabelIndices.AsSpan(0, _attribute));

                for (int i = 0; i < _attribute; i++)
                {
                    float bestConfidence = bestConfidences[i];

                    // Stop early if confidence is low
                    if (bestConfidence < confidenceThreshold)
                        continue;

                    int bestLabelIndex = bestLabelIndices[i];

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
                        xMin = (int)MathF.Round((x - w / 2 - xPad) * xGain);
                        yMin = (int)MathF.Round((y - h / 2 - yPad) * yGain);
                        xMax = (int)MathF.Round((x + w / 2 - xPad) * xGain);
                        yMax = (int)MathF.Round((y + h / 2 - yPad) * yGain);
                    }
                    else
                    {
                        var halfW = w / 2;
                        var halfH = h / 2;

                        // Manual clamping is faster than Math.Clamp in tight loops

                        int val = (int)MathF.Round((x - halfW - xPad) / xGain);
                        xMin = val < 0 ? 0 : (val > width - 1 ? width - 1 : val);

                        int valY = (int)MathF.Round((y - halfH - yPad) / yGain);
                        yMin = valY < 0 ? 0 : (valY > height - 1 ? height - 1 : valY);

                        int valXMax = (int)MathF.Round((x + halfW - xPad) / xGain);
                        xMax = valXMax < 0 ? 0 : (valXMax > width - 1 ? width - 1 : valXMax);

                        int valYMax = (int)MathF.Round((y + halfH - yPad) / yGain);
                        yMax = valYMax < 0 ? 0 : (valYMax > height - 1 ? height - 1 : valYMax);
                    }

                    var boundingBox = new SKRectI(xMin, yMin, xMax, yMax);

                    boxes[validBoxCount++] = new ObjectResult
                    {
                        Label = _yoloCore.OnnxModel.Labels[bestLabelIndex],
                        Confidence = bestConfidence,
                        BoundingBox = boundingBox,
                        BoundingBoxIndex = i,
                        OrientationAngle = _yoloCore.OnnxModel.ModelType == ModelType.ObbDetection ? ortSpan[i + _attribute * (4 + _labels)] : 0
                    };
                }

                // Get only the valid portion of the rented array
                var resultArray = boxes.AsSpan(0, validBoxCount);

                // Remove overlapping boxes using Non-Maximum Suppression (NMS)
                // Copy the results out: NMS compacts them inside 'boxes', which goes back to the shared ArrayPool
                // in the finally block below, where any other inference call could rent and overwrite it.
                return _yoloCore.RemoveOverlappingBoxes(resultArray, overlapThreshold).ToArray();
            }
            finally
            {
                ArrayPool<ObjectResult>.Shared.Return(boxes, false);
                ArrayPool<float>.Shared.Return(bestConfidences, false);
                ArrayPool<int>.Shared.Return(bestLabelIndices, false);
            }
        }

        /// <summary>
        /// Finds the highest label confidence, and the index of that label, for every bounding box.
        /// </summary>
        /// <remarks>
        /// The output tensor is laid out as [attributes, predictions], so the confidences of one box are
        /// <c>predictions</c> floats apart. Scanning label by label instead reads each label row sequentially
        /// and compares several boxes per instruction. As before, a box starts at confidence 0 with no label,
        /// and only a strictly higher confidence replaces the current best, so ties keep the first label.
        /// </remarks>
        private void FindBestLabels(ReadOnlySpan<float> ortSpan, Span<float> bestConfidences, Span<int> bestLabelIndices)
        {
            bestConfidences.Clear();
            bestLabelIndices.Fill(-1);

            var confidenceVectors = MemoryMarshal.Cast<float, Vector<float>>(bestConfidences);
            var labelVectors = MemoryMarshal.Cast<int, Vector<int>>(bestLabelIndices);
            int vectorizedCount = confidenceVectors.Length * Vector<float>.Count;

            for (int l = 0; l < _labels; l++)
            {
                var labelRow = ortSpan.Slice(_attribute4 + l * _attribute, _attribute);
                var rowVectors = MemoryMarshal.Cast<float, Vector<float>>(labelRow);
                var label = new Vector<int>(l);

                for (int v = 0; v < confidenceVectors.Length; v++)
                {
                    var isHigher = Vector.GreaterThan(rowVectors[v], confidenceVectors[v]);
                    confidenceVectors[v] = Vector.ConditionalSelect(isHigher, rowVectors[v], confidenceVectors[v]);
                    labelVectors[v] = Vector.ConditionalSelect(isHigher, label, labelVectors[v]);
                }

                // Scalar tail for the boxes that don't fill a whole vector
                for (int i = vectorizedCount; i < _attribute; i++)
                {
                    if (labelRow[i] > bestConfidences[i])
                    {
                        bestConfidences[i] = labelRow[i];
                        bestLabelIndices[i] = l;
                    }
                }
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
