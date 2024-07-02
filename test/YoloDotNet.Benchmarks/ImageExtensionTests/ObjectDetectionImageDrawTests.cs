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
            _cpuYolo = new Yolo(SharedConfig.GetTestModel(ModelType.ObjectDetection), false);
            _image = SKImage.FromEncodedData(SharedConfig.GetTestImage(ImageType.Street));
            _objectDetections = _cpuYolo.RunObjectDetection(_image);
        }

        [GlobalCleanup]
        public void Cleanup()
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
