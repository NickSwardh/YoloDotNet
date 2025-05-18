namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class SegmentationImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModelV8(ModelType.Segmentation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.People);

        private Yolo _cpuYolo;
        private SKBitmap _image;
        private List<Segmentation> _segmentations;

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
            _segmentations = _cpuYolo.RunSegmentation(_image);
        }

        [GlobalCleanup]
        public void CleanUp()
        {
            _cpuYolo.Dispose();
            _image.Dispose();
        }

        [Params(DrawSegment.Default, DrawSegment.PixelMaskOnly, DrawSegment.BoundingBoxOnly)]
        public DrawSegment DrawSegmentType { get; set; }

        [Benchmark]
        public void DrawSegmentation()
        {
            _image.Draw(_segmentations, DrawSegmentType);
        }

        #endregion Methods
    }
}
