namespace YoloDotNet.Benchmarks.ClassificationTests
{
    [MemoryDiagnoser]
    public class SimpleClassificationTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.Classification);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Hummingbird);

        private Yolo _cudaYolo;
        private Yolo _cpuYolo;
        private SKImage _image;
        private SKImage _skiaImage;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cudaYolo = new Yolo(onnxModel: _model, cuda: true, true);
            //_cpuYolo = new Yolo(onnxModel: _model, cuda: false);
            _image = SKImage.FromEncodedData(_testImage);
            _skiaImage = SKImage.FromEncodedData(_testImage);
        }

        //[Benchmark]
        //public void RunSimpleClassificationGpu()
        //{
        //    _ = _cudaYolo.RunClassification(img: _image, classes: 1);
        //}

        //[Benchmark]
        //public List<Classification> RunSimpleClassificationCpu()
        //{
        //    return _cpuYolo.RunClassification(img: _image, classes: 1);
        //}

        //[Benchmark]
        //public void RunSimpleObjectDetectionSkiaGpu()
        //{
        //    _ = _cudaYolo.RunClassification(img: _skiaImage, classes: 1);
        //}

        #endregion Methods
    }
}
