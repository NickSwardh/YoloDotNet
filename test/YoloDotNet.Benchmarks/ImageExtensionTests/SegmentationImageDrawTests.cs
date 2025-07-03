namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class SegmentationImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModelV8(ModelType.Segmentation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.People);

        private Yolo _cpuYolo;
        private SKBitmap _skBitmap;
        private SKImage _skImage;
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
            _skBitmap = SKBitmap.Decode(_testImage);
            _skImage = SKImage.FromEncodedData(_testImage);

            // We just need one result to use for drawing.
            _segmentations = _cpuYolo.RunSegmentation(_skBitmap);
        }

        [GlobalCleanup]
        public void CleanUp()
        {
            _cpuYolo?.Dispose();
            _skBitmap?.Dispose();
            _skImage?.Dispose();
        }

        /// <summary>
        /// Draw on an SKImage.
        /// </summary>
        [Benchmark]
        public void DrawSegmentationOnSkImage()
        {
            // When drawing using an SKimage, a new SKBitmap is returned with the drawn objects.
            _ = _skImage.Draw(_segmentations);
        }

        /// <summary>
        /// Draw on an SKBitmap.
        /// </summary>
        [Benchmark]
        public void DrawSegmentationOnSKBitmap()
        {
            _skBitmap.Draw(_segmentations);
        }

        #endregion Methods
    }
}
