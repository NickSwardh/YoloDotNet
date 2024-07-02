namespace YoloDotNet.Benchmarks.PoseEstimationTests
{
    [MemoryDiagnoser]
    public class SimplePostEstimationTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModel(ModelType.PoseEstimation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

        private Yolo _cudaYolo;
        private Yolo _cpuYolo;
        private SKImage _image;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cudaYolo = new Yolo(_model, true);
            _cpuYolo = new Yolo(_model, false);
            _image = SKImage.FromEncodedData(_testImage);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cpuYolo.Dispose();
            _cudaYolo.Dispose();
            _image.Dispose();
        }

        [Benchmark]
        public List<PoseEstimation> RunSimplePoseEstimationGpu()
        {
            return _cudaYolo.RunPoseEstimation(_image);
        }

        [Benchmark]
        public List<PoseEstimation> RunSimplePoseEstimationCpu()
        {
            return _cpuYolo.RunPoseEstimation(_image);
        }

        #endregion Methods
    }
}
