namespace YoloDotNet.Benchmarks.PoseEstimationTests
{
    [MemoryDiagnoser]
    public class SimplePoseEstimationTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModelV8(ModelType.PoseEstimation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

        private Yolo _cudaYolo;
        private Yolo _cpuYolo;
        private SKImage _image;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            var options = new YoloOptions
            {
                OnnxModel = _model,
                ModelType = ModelType.PoseEstimation,
                Cuda = true
            };

            _cudaYolo = new Yolo(options);

            options.Cuda = false;
            _cpuYolo = new Yolo(options);
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
        public List<PoseEstimation> RunSimplePoseEstimationCpu()
        {
            return _cpuYolo.RunPoseEstimation(_image);
        }

        [Benchmark]
        public List<PoseEstimation> RunSimplePoseEstimationGpu()
        {
            return _cudaYolo.RunPoseEstimation(_image);
        }

        #endregion Methods
    }
}
