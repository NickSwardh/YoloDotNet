using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Memory;

namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class ResizeImageTests
    {
        #region Fields

        private static string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private Image _imageSharp;
        private SKImage _image;
        private readonly int _width = 240;
        private readonly int _height = 240;
        private SKImageInfo _imageInfo;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            //_imageSharp = Image.Load<Rgba32>(_testImage);
            //_image = SKImage.FromEncodedData(_testImage);
            _imageInfo = new SKImageInfo(_width, _height, SKColorType.Rgb888x, SKAlphaType.Opaque);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            //_imageSharp.Dispose();
            //_image.Dispose();
            
        }

        //[Benchmark]
        //public ReadOnlySpan<byte> ResizeImage()
        //{
        //    return _image.ResizeImage(_cpuYolo.OnnxModel.Input.Width, _cpuYolo.OnnxModel.Input.Height);
        //}

        [Benchmark]
        public void ResizeWithImageSharp()
        {
            using var image = Image.Load<Rgba32>(_testImage);

            var options = new ResizeOptions
            {
                Size = new Size(_width, _height),
                Mode = ResizeMode.Pad,
                PadColor = new Color(new Rgb24(0, 0, 0))
            };

            _ = image.Clone(x => x.Resize(options)).CloneAs<Rgb24>();
        }

        [Benchmark]
        public void ResizeWithSkiaSharp()
        {
            using var image = SKImage.FromEncodedData(_testImage);
            _ = image.ResizeImage(_imageInfo);
        }

        //[Benchmark]
        //public void ResizeImageNew()
        //{
        //    using var image = SKImage.FromEncodedData(_testImage);
        //    _ = image.ResizeImageNew(_width, _height);
        //}

        #endregion Methods
    }
}
