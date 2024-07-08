namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class SegmentationImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModel(ModelType.Segmentation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.People);

        private Yolo _cpuYolo;
        private SKImage _image;
        private List<Segmentation> _segmentations;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            var options = new YoloOptions
            {
                OnnxModel = _model,
                ModelType = ModelType.Segmentation,
                Cuda = false
            };

            _cpuYolo = new Yolo(options);
            _image = SKImage.FromEncodedData(_testImage);
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
        public SKImage DrawSegmentation()
        {
            return _image.Draw(_segmentations, DrawSegmentType);
        }

        #endregion Methods
    }
}
