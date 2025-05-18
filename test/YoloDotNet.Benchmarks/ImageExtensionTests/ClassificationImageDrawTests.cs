namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class ClassificationImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModelV8(ModelType.Classification);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private Yolo _cpuYolo;
        private SKImage _skImage;
        private SKBitmap _skBitmap;
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
            _skBitmap = SKBitmap.Decode(_testImage);
            _skImage = SKImage.FromEncodedData(_testImage);

            // We just need one result to use for drawing.
            _classifications = _cpuYolo.RunClassification(_skBitmap);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cpuYolo?.Dispose();
            _skImage?.Dispose();
            _skBitmap?.Dispose();
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public void DrawClassificationOnSKImage()
        {
            _skImage.Draw(_classifications, DrawConfidence);
        }

        [Benchmark]
        public void DrawClassificationOnSKBitmap()
        {
            _skBitmap.Draw(_classifications, DrawConfidence);
        }

        #endregion Methods
    }
}
