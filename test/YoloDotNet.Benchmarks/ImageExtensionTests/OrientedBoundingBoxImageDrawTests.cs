namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class OrientedBoundingBoxImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModel(ModelType.ObbDetection);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Island);

        private Yolo _cpuYolo;
        private SKImage _image;
        private List<OBBDetection> _oBBDetections;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cpuYolo = new Yolo(_model, false);
            _image = SKImage.FromEncodedData(_testImage);
            _oBBDetections = _cpuYolo.RunObbDetection(_image);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _cpuYolo.Dispose();
            _image.Dispose();
        }

        [Benchmark]
        public SKImage DrawOrientedBoundingBox()
        {
            return _image.Draw(_oBBDetections);
        }

        #endregion Methods
    }
}
