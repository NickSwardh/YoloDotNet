namespace YoloDotNet.Modules
{
    internal class PoseEstimationModule
        : IDetectionModule, IModule<List<PoseEstimation>, Dictionary<int, List<PoseEstimation>>>
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModule _objectDetectionModule;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public PoseEstimationModule(string onnxModel, bool cuda = true, bool primeGpu = false, int gpuId = 0)
        {
            _yoloCore = new YoloCore(onnxModel, cuda, primeGpu, gpuId);
            _objectDetectionModule = new ObjectDetectionModule(_yoloCore);
            _yoloCore.InitializeYolo(ModelType.PoseEstimation);
            SubscribeToVideoEvents();
        }

        public List<PoseEstimation> ProcessImage(SKImage image, double confidence, double iou)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(image);
            using var ort = ortValues[0];
            return PoseEstimateImage(image, ort, confidence, iou);
        }

        public Dictionary<int, List<PoseEstimation>> ProcessVideo(VideoOptions options, double confidence, double iou)
            => _yoloCore.RunVideo(options, confidence, iou, ProcessImage);

        #region Helper methods

        public List<PoseEstimation> PoseEstimateImage(SKImage image, OrtValue ortTensor, double threshold, double overlapThrehshold)
        {
            var boxes = _objectDetectionModule.ObjectDetectImage(image, ortTensor, threshold, overlapThrehshold);
            //var boxes = ObjectDetectImage(image, threshold, overlapThrehshold);


            var (xPad, yPad, gain) = _yoloCore.CalculateGain(image);

            // Get tensor as a flattened Span for faster processing.
            var ortSpan = ortTensor.GetTensorDataAsSpan<float>();

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

            return boxes.Select(x => (PoseEstimation)x).ToList();
        }

        private void SubscribeToVideoEvents()
        {
            _yoloCore.VideoProgressEvent += (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _yoloCore.VideoCompleteEvent += (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _yoloCore.VideoStatusEvent += (sender, e) => VideoStatusEvent?.Invoke(sender, e);
        }

        public void Dispose()
        {
            _yoloCore.VideoProgressEvent -= (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _yoloCore.VideoCompleteEvent -= (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _yoloCore.VideoStatusEvent -= (sender, e) => VideoStatusEvent?.Invoke(sender, e);

            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
