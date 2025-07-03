namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class OrientedBoundingBoxImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModelV8(ModelType.ObbDetection);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Island);

        private Yolo _cpuYolo;
        private SKImage _skImage;
        private SKBitmap _skBitmap;
        private List<OBBDetection> _oBBDetections;

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
            _oBBDetections = _cpuYolo.RunObbDetection(_skBitmap);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cpuYolo?.Dispose();
            _skBitmap?.Dispose();
            _skImage?.Dispose();
        }

        [Params(false, true)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public void DrawOrientedBoundingBoxOnSKImage()
        {
            // When drawing using an SKimage, a new SKBitmap is returned with the drawn objects.
            _ = _skImage.Draw(_oBBDetections);
        }

        [Benchmark]
        public void DrawOrientedBoundingBoxONSKBitmap()
        {
            _skBitmap.Draw(_oBBDetections);
        }

        #endregion Methods
    }
}
