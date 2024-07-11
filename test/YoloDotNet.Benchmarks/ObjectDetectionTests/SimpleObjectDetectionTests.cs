namespace YoloDotNet.Benchmarks.ObjectDetectionTests
{
    [MemoryDiagnoser]
    public class SimpleObjectDetectionTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModel(ModelType.ObjectDetection);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);

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
                ModelType = ModelType.ObjectDetection,
                Cuda = true
            };

            _cudaYolo = new Yolo(options);

            options.Cuda = false;
            _cpuYolo = new Yolo(options);
            _image = SKImage.FromEncodedData(_testImage);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _cudaYolo.Dispose();
            _cpuYolo.Dispose();
            _image.Dispose();
        }

        [Benchmark]
        public List<ObjectDetection> RunSimpleObjectDetectionCpu()
        {
            return _cpuYolo.RunObjectDetection(_image);
        }

        [Benchmark]
        public List<ObjectDetection> RunSimpleObjectDetectionGpu()
        {
            return _cudaYolo.RunObjectDetection( _image);
        }

        #endregion Methods
    }
}
