using Microsoft.VSDiagnostics;

namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[MemoryDiagnoser]
    [CPUUsageDiagnoser]
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

        [Params(false, true)]
        public bool DrawConfidence { get; set; }


        [Benchmark(Baseline = true)]
        public void DrawObjectDetectionOnSKImage()
        {
            // When drawing using an SKimage, a new SKBitmap is returned with the drawn objects.
            _ = _skImage.Draw(_objectDetections, DrawConfidence);
        }

        [Benchmark(Baseline = true)]
        public void DrawObjectDetectionOnSKBitmap()
        {
            _skBitmap.Draw(_objectDetections, DrawConfidence);
        }

        #endregion Methods
    }
}
