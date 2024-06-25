namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class ResizeImageTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.ObjectDetection);
        private static string _testImage = SharedConfig.GetTestImage(imageName: "street640x640.jpg");

        private Yolo _cpuYolo;
        private Image _image;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cpuYolo = new Yolo(onnxModel: _model, cuda: false);
            _image = Image.Load(path: _testImage);
        }

        [Benchmark]
        public Image<Rgb24> ResizeImage()
        {
            return _image.ResizeImage(
                        w: _cpuYolo.OnnxModel.Input.BatchSize,
                        h: _cpuYolo.OnnxModel.Input.Channels);
        }

        #endregion Methods
    }
}
