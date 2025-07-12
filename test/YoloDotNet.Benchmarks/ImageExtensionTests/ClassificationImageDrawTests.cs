namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class ClassificationImageDrawTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private Yolo _yolo;
        private SKBitmap _skBitmap;
        private List<Classification> _classifications;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _yolo = YoloCreator.CreateYolo(YoloType.V8_Cls_CPU);
            _skBitmap = SKBitmap.Decode(_testImage);

            _classifications = _yolo.RunClassification(_skBitmap);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _yolo?.Dispose();
            _skBitmap?.Dispose();
        }

        [Benchmark]
        public void DrawClassification()
            => _skBitmap.Draw(_classifications);
    }
}
