namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class ObjectDetectionImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModel(ModelType.ObjectDetection);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);

        private Yolo _cpuYolo;
        private SKImage _image;
        private List<ObjectDetection> _objectDetections;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            var options = new YoloOptions
            {
                OnnxModel = _model,
                ModelType = ModelType.ObjectDetection,
                Cuda = false
            };

            _cpuYolo = new Yolo(options);
            _image = SKImage.FromEncodedData(_testImage);
            _objectDetections = _cpuYolo.RunObjectDetection(_image);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cpuYolo.Dispose();
            _image.Dispose();
        }

        [Params(false, true)]
        public bool DrawConfidence { get; set; }


        [Benchmark(Baseline = true)]
        public SKImage DrawObjectDetection()
        {
            return _image.Draw(_objectDetections, DrawConfidence);
        }

        #endregion Methods
    }
}
