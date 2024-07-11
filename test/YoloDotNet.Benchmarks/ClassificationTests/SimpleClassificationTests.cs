namespace YoloDotNet.Benchmarks.ClassificationTests
{
    [MemoryDiagnoser]
    public class SimpleClassificationTests
    {
        #region Fields

        private static readonly string _model = SharedConfig.GetTestModel(ModelType.Classification);
        private static readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

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
                ModelType = ModelType.Classification
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
        public void RunSimpleClassificationCpu()
        {
            _ = _cpuYolo.RunClassification(_image);
        }

        [Benchmark]
        public void RunSimpleClassificationGpu()
        {
            _ = _cudaYolo.RunClassification(_image);
        }


        #endregion Methods
    }
}
