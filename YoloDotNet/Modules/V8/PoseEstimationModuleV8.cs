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

        public List<PoseEstimation> ProcessImage(SKImage image, double confidence, double pixelConfidence, double iou)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(image);
            var ortSpan = ortValues[0].GetTensorDataAsSpan<float>(); ;

            return PoseEstimateImage(image, ortSpan, confidence, iou);

        }

        #region Helper methods

        public List<PoseEstimation> PoseEstimateImage(SKImage image, ReadOnlySpan<float> ortSpan, double threshold, double overlapThrehshold)
        {
            var boxes = _objectDetectionModule.ObjectDetection(image, ortSpan, threshold, overlapThrehshold);

            // TODO: Implement for stretched input images too.
            var (xPad, yPad, gain, _) = _yoloCore.CalculateGain(image);

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

                    var x = (int)((ortSpan[xIndex] - xPad) * gain);
                    var y = (int)((ortSpan[yIndex] - yPad) * gain);
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
