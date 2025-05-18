namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class ClassificationImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModelV8(ModelType.Classification);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private Yolo _cpuYolo;
        private SKBitmap _image;
        private List<Classification> _classifications;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            var options = new YoloOptions
            {
                OnnxModel = _model,
                Cuda = false
            };

            _cpuYolo = new Yolo(options);
            _image = SKBitmap.Decode(_testImage);
            _classifications = _cpuYolo.RunClassification(_image);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cpuYolo.Dispose();
            _image.Dispose();
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public void DrawClassification()
        {
            _image.Draw(_classifications, DrawConfidence);
        }

        #endregion Methods
    }
}
