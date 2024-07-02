namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class ClassificationImageDrawTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(ModelType.Classification);
        private static string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private Yolo _cpuYolo;
        private SKImage _image;
        private List<Classification> _classifications;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cpuYolo = new Yolo(_model, false);
            _image = SKImage.FromEncodedData(_testImage);
            _classifications = _cpuYolo.RunClassification(_image);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _cpuYolo.Dispose();
            _image.Dispose();
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public SKImage DrawClassification()
        {
            return _image.Draw(classifications: _classifications, drawConfidence: DrawConfidence);
        }

        #endregion Methods
    }
}
