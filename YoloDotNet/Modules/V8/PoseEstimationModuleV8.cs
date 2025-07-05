namespace YoloDotNet.Modules.V8
{
    internal class PoseEstimationModuleV8 : IPoseEstimationModule
    {
        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public PoseEstimationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);
        }

        public List<PoseEstimation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
        {
            var (ortValues, imageSize) = _yoloCore.Run(image);
            using IDisposableReadOnlyCollection<OrtValue> _ = ortValues;

            var ortSpan = ortValues[0].GetTensorDataAsSpan<float>();

            return PoseEstimateImage(imageSize, ortSpan, confidence, iou);

        }

        #region Helper methods

        public List<PoseEstimation> PoseEstimateImage(SKSizeI imageSize, ReadOnlySpan<float> ortSpan, double threshold, double overlapThrehshold)
        {
            var boxes = _objectDetectionModule.ObjectDetection(imageSize, ortSpan, threshold, overlapThrehshold);

            var (xPad, yPad, gain, _) = _yoloCore.CalculateGain(imageSize);

            var labels = _yoloCore.OnnxModel.Labels.Length;
            var ouputChannels = _yoloCore.OnnxModel.Outputs[0].Channels;
            var totalKeypoints = (int)Math.Floor(((double)_yoloCore.OnnxModel.Outputs[0].Elements / _yoloCore.OnnxModel.Input.Channels)) - labels;

            for (int i = 0; i < boxes.Length; i++)
            {
                var box = boxes[i];
                var poseEstimations = new KeyPoint[totalKeypoints];
                var keypointOffset = box.BoundingBoxIndex + (ouputChannels * (4 + labels)); // Skip boundingbox + labels (4 + labels) and move forward to the first keypoint

                for (var j = 0; j < totalKeypoints; j++)
                {
                    var xIndex = keypointOffset;
                    var yIndex = xIndex + ouputChannels;
                    var cIndex = yIndex + ouputChannels;
                    keypointOffset += ouputChannels * 3;

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

            return [.. boxes.Select(x => (PoseEstimation)x)];
        }

        public void Dispose()
        {
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
