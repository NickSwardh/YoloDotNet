namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class OrientedBoundingBoxImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModelV8(ModelType.ObbDetection);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Island);

        private Yolo _cpuYolo;
        private SKBitmap _image;
        private List<OBBDetection> _oBBDetections;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            var options = new YoloOptions
            {
                OnnxModel = _model,
                ModelType = ModelType.ObbDetection,
                Cuda = false
            };

            _cpuYolo = new Yolo(options);
            _image = SKBitmap.Decode(_testImage);
            _oBBDetections = _cpuYolo.RunObbDetection(_image);
        }
        
        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cpuYolo?.Dispose();
            _image?.Dispose();
        }

        [Params(false, true)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public void DrawOrientedBoundingBox()
        {
            _image.Draw(_oBBDetections, DrawConfidence);
        }

        #endregion Methods
    }
}
