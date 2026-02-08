// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V8
{
    internal class PoseEstimationModuleV8 : IPoseEstimationModule
    {
        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule;
        private int _inputChannels;
        private int _modelOutputChannels;
        private int _modelOutputElements;
        private List<PoseEstimation> _results = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public PoseEstimationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Get input shape from ONNX model. Format NCHW: [Batch (B), Channels (C), Height (H), Width (W)]
            var inputShape = _yoloCore.OnnxModel.InputShapes.ElementAt(0).Value;

            // Get output shape from ONNX model. Format: [Batch, Attributes, Predictions]
            var outputShape = _yoloCore.OnnxModel.OutputShapes.ElementAt(0).Value;

            _inputChannels = (int)inputShape[1];
            _modelOutputElements = outputShape[1];
            _modelOutputChannels = outputShape[2];

            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);
            _results = [];
        }

        public List<PoseEstimation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            var inferenceResult = _yoloCore.Run(image, roi);
            var detections = PoseEstimateImage(inferenceResult, confidence, iou);

            return YoloCore.InferenceResultsToType(detections, roi, _results, r => (PoseEstimation)r);
        }

        #region Helper methods

        private Span<ObjectResult> PoseEstimateImage(InferenceResult inferenceResult, double threshold, double overlapThrehshold)
        {
            var boxes = _objectDetectionModule.ObjectDetection(inferenceResult, threshold, overlapThrehshold);

            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            var (xPad, yPad, gain, _) = _yoloCore.CalculateGain(imageSize);

            var labels = _yoloCore.OnnxModel.Labels.Length;
            var totalKeypoints = (int)Math.Floor(((double)_modelOutputElements / _inputChannels)) - labels;

            var totalBoxes = boxes.Length;
            for (int i = 0; i < totalBoxes; i++)
            {
                var box = boxes[i];
                var poseEstimations = new KeyPoint[totalKeypoints];
                var keypointOffset = box.BoundingBoxIndex + (_modelOutputChannels * (4 + labels)); // Skip boundingbox + labels (4 + labels) and move forward to the first keypoint

                for (var j = 0; j < totalKeypoints; j++)
                {
                    var xIndex = keypointOffset;
                    var yIndex = xIndex + _modelOutputChannels;
                    var cIndex = yIndex + _modelOutputChannels;
                    keypointOffset += _modelOutputChannels * 3;

                    var x = 0;
                    var y = 0;

                    if (_yoloCore.YoloOptions.ImageResize == ImageResize.Proportional)
                    {
                        x = (int)((ortSpan[xIndex] - xPad) * gain);
                        y = (int)((ortSpan[yIndex] - yPad) * gain);
                    }
                    else
                    {
                        // Map keypoints proportionally into resized bounding box
                        var relativeX = (ortSpan[xIndex] - box.BoundingBoxUnscaled.Left) / box.BoundingBoxUnscaled.Width;
                        var relativeY = (ortSpan[yIndex] - box.BoundingBoxUnscaled.Top) / box.BoundingBoxUnscaled.Height;

                        x = (int)(box.BoundingBox.Left + relativeX * box.BoundingBox.Width);
                        y = (int)(box.BoundingBox.Top + relativeY * box.BoundingBox.Height);

                        // Clamp to ensure keypoint is inside box
                        x = Math.Clamp(x, box.BoundingBox.Left, box.BoundingBox.Right - 1);
                        y = Math.Clamp(y, box.BoundingBox.Top, box.BoundingBox.Bottom - 1);
                    }

                    var confidence = ortSpan[cIndex];

                    poseEstimations[j] = new KeyPoint(x, y, confidence);
                }

                box.KeyPoints = poseEstimations;
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
