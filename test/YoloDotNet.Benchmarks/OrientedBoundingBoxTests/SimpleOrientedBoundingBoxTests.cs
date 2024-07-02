namespace YoloDotNet.Benchmarks.OrientedBoundingBoxTests
{
    [MemoryDiagnoser]
    public class SimpleOrientedBoundingBoxTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModel(ModelType.ObbDetection);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Island);

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
            _cudaYolo.Dispose();
            _cpuYolo.Dispose();
            _image.Dispose();
        }

        [Benchmark]
        public List<OBBDetection> RunSimpleObbDetectionGpu()
        {
            return _cudaYolo.RunObbDetection(_image);
        }

        [Benchmark]
        public List<OBBDetection> RunSimpleObbDetectionCpu()
        {
            return _cpuYolo.RunObbDetection(_image);
        }

        #endregion Methods
    }
}
