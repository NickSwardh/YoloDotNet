namespace YoloDotNet.Benchmarks.ObjectDetectionTests
{
    [MemoryDiagnoser]
    public class SimpleObjectDetectionTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModel(modelType: ModelType.ObjectDetection);
        private readonly string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Street);

        private Yolo _cudaYolo;
        private Yolo _cpuYolo;
        private SKImage _image;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cudaYolo = new Yolo(onnxModel: _model, cuda: true, true);
            _cpuYolo = new Yolo(onnxModel: _model, cuda: false);
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
        public List<ObjectDetection> RunSimpleObjectDetectionGpu()
        {
            return _cudaYolo.RunObjectDetection( _image);
        }

        [Benchmark]
        public List<ObjectDetection> RunSimpleObjectDetectionCpu()
        {
            return _cpuYolo.RunObjectDetection(_image);
        }

        #endregion Methods
    }
}
