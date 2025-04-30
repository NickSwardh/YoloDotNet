namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class ResizeImageTests
    {
        #region Fields

        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private SKBitmap _image;
        private SKImageInfo _outputImageInfo;
        private readonly int _width = 240;
        private readonly int _height = 240;

        // Best quality - Uses cubic resamplers for smoother, high-quality scaling  
        // - ✅ Produces sharp, artifact-free images  
        // - 🔄 Slightly slower than linear or nearest filtering  
        private readonly SKSamplingOptions _cubicMitchell = new(SKCubicResampler.Mitchell);
        private readonly SKSamplingOptions _cubicCatmullRom = new(SKCubicResampler.CatmullRom);

        // Balanced quality and performance - Uses linear filtering with different mipmap strategies  
        // - ✅ Good balance of sharpness and speed  
        // - 🔄 Mipmaps improve quality when downscaling, but may slightly impact performance  
        private readonly SKSamplingOptions _linearWithLinearMipmap = new(SKFilterMode.Linear, SKMipmapMode.Linear);
        private readonly SKSamplingOptions _linearWithNearestMipmap = new(SKFilterMode.Linear, SKMipmapMode.Nearest);
        private readonly SKSamplingOptions _linearNoMipmap = new(SKFilterMode.Linear, SKMipmapMode.None);               // Default in YoloDotNet  

        // Fastest performance, but lower quality - Uses nearest-neighbor filtering  
        // - ✅ Very fast, best for real-time performance  
        // - ❌ Can introduce pixelation and jagged edges  
        private readonly SKSamplingOptions _nearestWithLinearMipmap = new(SKFilterMode.Nearest, SKMipmapMode.Linear);
        private readonly SKSamplingOptions _nearestWithNearestMipmap = new(SKFilterMode.Nearest, SKMipmapMode.Nearest);
        private readonly SKSamplingOptions _nearestNoMipmap = new(SKFilterMode.Nearest, SKMipmapMode.None);

        // Anisotropic filtering - Improves texture quality at oblique angles  
        // - ✅ Reduces blurriness and distortion on angled surfaces  
        // - 🔄 Higher levels (8x, 16x) improve quality but may reduce performance  
        private readonly SKSamplingOptions _anisotropic4x = new(4);  // Moderate improvement  
        private readonly SKSamplingOptions _anisotropic8x = new(8);  // Higher quality, slight performance cost  
        private readonly SKSamplingOptions _anisotropic16x = new(16); // Best for extreme angles, highest performance cost  

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _outputImageInfo = new SKImageInfo(_width, _height, SKColorType.Rgb888x, SKAlphaType.Opaque);
            _image = SKBitmap.Decode(_testImage);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _image.Dispose();
        }

        /// <summary>
        /// Benchmark for resizing using Cubic Mitchell resampler.
        /// - **Quality:** High (smooth results, reduced aliasing).
        /// - **Performance:** Moderate.
        /// - **Best Use Case:** When high-quality resampling is needed.
        /// </summary>
        [Benchmark]
        public void ResizeImage_CubicMitchel()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo, _cubicMitchell);
        }

        /// <summary>
        /// Benchmark for resizing using Cubic Catmull-Rom resampler.
        /// - **Quality:** Very sharp (preserves details well but can introduce ringing artifacts).
        /// - **Performance:** Moderate.
        /// - **Best Use Case:** When sharpness is more important than smoothness.
        /// </summary>
        [Benchmark]
        public void ResizeImage_CubicCatmullRom()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo, _cubicCatmullRom);
        }

        /// <summary>
        /// Benchmark for resizing using Linear filtering with Linear Mipmap interpolation.
        /// - **Quality:** Balanced (smooth with slight blur).
        /// - **Performance:** Good.
        /// - **Best Use Case:** When scaling down with a balance of speed and quality.
        /// </summary>
        [Benchmark]
        public void ResizeImage_LinearWithLinearMipmap()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo, _linearWithLinearMipmap);
        }

        /// <summary>
        /// Benchmark for resizing using Linear filtering with Nearest Mipmap interpolation.
        /// - **Quality:** Slightly sharper than Linear Mipmap.
        /// - **Performance:** Good.
        /// - **Best Use Case:** When performance is slightly prioritized over smoothness.
        /// </summary>
        [Benchmark]
        public void ResizeImage_LinearWithNearestMipmap()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo, _linearWithNearestMipmap);
        }

        /// <summary>
        /// Benchmark for resizing using Linear filtering with no Mipmap. (Default in YoloDotNet) 
        /// - **Quality:** Decent (default option, slight blur).
        /// - **Performance:** Fast.
        /// - **Best Use Case:** General-purpose resizing with a balance of speed and quality.
        /// </summary>
        [Benchmark]
        public void ResizeImage_LinearNoMipmap()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo, _linearNoMipmap);
        }

        /// <summary>
        /// Benchmark for resizing using Nearest filtering with Linear Mipmap interpolation.
        /// - **Quality:** Lower (visible pixelation).
        /// - **Performance:** Very fast.
        /// - **Best Use Case:** When maximum speed is needed but mipmaps help reduce aliasing.
        /// </summary>
        [Benchmark]
        public void ResizeImage_NearestWithLinearMipmap()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo, _nearestWithLinearMipmap);
        }

        /// <summary>
        /// Benchmark for resizing using Nearest filtering with Nearest Mipmap interpolation.
        /// - **Quality:** Low (pixelated, harsh transitions).
        /// - **Performance:** Very fast.
        /// - **Best Use Case:** When raw performance is the top priority.
        /// </summary>
        [Benchmark]
        public void ResizeImage_NearestWithNearestMipmap()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo, _nearestWithNearestMipmap);
        }

        /// <summary>
        /// Benchmark for resizing using Nearest filtering with no Mipmap.
        /// - **Quality:** Lowest (hard edges, pixelation).
        /// - **Performance:** Fastest.
        /// - **Best Use Case:** When performance is critical, and quality is not a concern.
        /// </summary>
        [Benchmark]
        public void ResizeImage_NearestNoMipmap()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo, _nearestNoMipmap);
        }

        /// <summary>
        /// Benchmark for resizing using 4x Anisotropic filtering.
        /// - **Quality:** Good (reduces blurring on angled textures).
        /// - **Performance:** Moderate.
        /// - **Best Use Case:** When dealing with textures viewed at an angle.
        /// </summary>
        [Benchmark]
        public void ResizeImage_Anisotropic4x()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo, _anisotropic4x);
        }

        /// <summary>
        /// Benchmark for resizing using 8x Anisotropic filtering.
        /// - **Quality:** Very good (better sharpness on textures at an angle).
        /// - **Performance:** Lower than 4x but still efficient.
        /// - **Best Use Case:** When dealing with angled textures and need better detail.
        /// </summary>
        [Benchmark]
        public void ResizeImage_Anisotropic8x()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo, _anisotropic8x);
        }

        /// <summary>
        /// Benchmark for resizing using 16x Anisotropic filtering.
        /// - **Quality:** Best (preserves details on angled textures).
        /// - **Performance:** Slowest among anisotropic options.
        /// - **Best Use Case:** When the highest texture quality is required, especially for angled surfaces.
        /// </summary>
        [Benchmark]
        public void ResizeImage_Anisotropic16x()
        {
            _ = _image.ResizeImageProportional(_outputImageInfo, _anisotropic16x);
        }

        #endregion Methods
    }
}
