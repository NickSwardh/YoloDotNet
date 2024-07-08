namespace YoloDotNet.Benchmarks.SegmentationTests
{
    [MemoryDiagnoser]
    public class SimpleSegmentationTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModel(ModelType.Segmentation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.People);

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
                ModelType = ModelType.Segmentation,
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
            _cudaYolo.Dispose();
            _cpuYolo.Dispose();
            _image.Dispose();
        }

        [Benchmark]
        public List<Segmentation> RunSimpleSegmentationCpu()
        {
            return _cpuYolo.RunSegmentation(_image);
        }

        [Benchmark]
        public List<Segmentation> RunSimpleSegmentationGpu()
        {
            return _cudaYolo.RunSegmentation(_image);
        }

        #endregion Methods
    }
}
