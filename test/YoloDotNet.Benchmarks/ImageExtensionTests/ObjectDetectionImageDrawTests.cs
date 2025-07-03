namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class ObjectDetectionImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModelV8(ModelType.ObjectDetection);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);

        private Yolo _cpuYolo;
        private SKImage _skImage;
        private SKBitmap _skBitmap;
        private List<ObjectDetection> _objectDetections;

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
            _objectDetections = _cpuYolo.RunObjectDetection(_skBitmap);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cpuYolo?.Dispose();
            _skBitmap?.Dispose();
            _skImage?.Dispose();
        }

        [Benchmark]
        public void DrawObjectDetectionOnSKImage()
        {
            // When drawing using an SKimage, a new SKBitmap is returned with the drawn objects.
            _ = _skImage.Draw(_objectDetections);
        }

        [Benchmark]
        public void DrawObjectDetectionOnSKBitmap()
        {
            _skBitmap.Draw(_objectDetections);
        }

        #endregion Methods
    }
}
