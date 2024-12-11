namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class PoseEstimationImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModelV8(ModelType.PoseEstimation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

        private Yolo _cpuYolo;
        private SKImage _image;
        private List<PoseEstimation> _poseEstimations;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            var options = new YoloOptions
            {
                OnnxModel = _model,
                ModelType = ModelType.PoseEstimation,
                HwAccelerator = HwAcceleratorType.None
            };

            _cpuYolo = new Yolo(options);
            _image = SKImage.FromEncodedData(_testImage);
            _poseEstimations = _cpuYolo.RunPoseEstimation(_image);
        }

        [GlobalCleanup]
        public void CleanUp()
        {
            _cpuYolo.Dispose();
            _image.Dispose();
        }

        [Params(true, false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public SKImage DrawPoseEstimation()
        {
            return _image.Draw(_poseEstimations, CustomKeyPointColorMap.KeyPointOptions, DrawConfidence);
        }

        #endregion Methods
    }
}
