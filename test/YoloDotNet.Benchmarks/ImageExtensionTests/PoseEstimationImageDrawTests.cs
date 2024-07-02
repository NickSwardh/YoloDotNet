namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class PoseEstimationImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModel(modelType: ModelType.PoseEstimation);
        private readonly string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Crosswalk);

        private Yolo _cpuYolo;
        private SKImage _image;
        private List<PoseEstimation> _poseEstimations;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cpuYolo = new Yolo(_model, false);
            _image = SKImage.FromEncodedData(_testImage);
            _poseEstimations = _cpuYolo.RunPoseEstimation(_image);
        }

        [GlobalCleanup]
        public void CleanUp()
        {
            _cpuYolo.Dispose();
            _image.Dispose();
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public SKImage DrawPoseEstimation()
        {
            return _image.Draw(_poseEstimations, CustomKeyPointColorMap.KeyPointOptions, DrawConfidence);
        }

        #endregion Methods
    }
}
