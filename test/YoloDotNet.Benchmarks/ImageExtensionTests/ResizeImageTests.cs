namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class ResizeImageTests
    {
        #region Fields

        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private SKImage _image;
        private SKImageInfo _outputImageInfo;
        private readonly int _width = 240;
        private readonly int _height = 240;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _outputImageInfo = new SKImageInfo(_width, _height, SKColorType.Rgb888x, SKAlphaType.Opaque);
            _image = SKImage.FromEncodedData(_testImage);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _image.Dispose();
        }

        [Benchmark]
        public void ResizeImage()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo);
        }

        #endregion Methods
    }
}
