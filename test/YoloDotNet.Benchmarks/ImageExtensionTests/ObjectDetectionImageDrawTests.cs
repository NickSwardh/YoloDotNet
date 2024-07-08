namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class ObjectDetectionImageDrawTests
    {
        #region Fields

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
                ModelType = ModelType.ObjectDetection,
                OnnxModel = SharedConfig.GetTestModel(ModelType.ObjectDetection),
                Cuda = false
            };

            _cpuYolo = new Yolo(options);
            _image = SKImage.FromEncodedData(SharedConfig.GetTestImage(ImageType.Street));
            _objectDetections = _cpuYolo.RunObjectDetection(_image);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cpuYolo.Dispose();
            _image.Dispose();
        }

        [Benchmark(Baseline = true)]
        public SKImage DrawObjectDetection()
        {
            return _image.Draw(_objectDetections);
        }

        #endregion Methods
    }
}
