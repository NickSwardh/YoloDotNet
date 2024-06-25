namespace YoloDotNet.Benchmarks.ObjectDetectionTests
{
    [MemoryDiagnoser]
    public class SimpleObjectDetectionTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.ObjectDetection);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Street);

        private Yolo _cudaYolo;
        private Yolo _cpuYolo;
        private Image _image;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cudaYolo = new Yolo(onnxModel: _model, cuda: true);
            _cpuYolo = new Yolo(onnxModel: _model, cuda: false);
            _image = Image.Load<Rgba32>(path: _testImage);
        }

        [Benchmark]
        public List<ObjectDetection> RunSimpleObjectDetectionGpu()
        {
            return _cudaYolo.RunObjectDetection(img: _image, confidence: 0.25, iou: 0.45);
        }

        [Benchmark]
        public List<ObjectDetection> RunSimpleObjectDetectionCpu()
        {
            return _cpuYolo.RunObjectDetection(img: _image, confidence: 0.25, iou: 0.45);
        }

        #endregion Methods
    }
}
