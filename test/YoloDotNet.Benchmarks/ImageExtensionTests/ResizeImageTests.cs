// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class ResizeImageTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private SKBitmap _image;

        private SKImageInfo _outputImageInfo;
        private PinnedMemoryBufferPool _pinnedBufferPool;
        private readonly int _width = 240;
        private readonly int _height = 240;

        public enum SamplingProfile
        {
            CubicMitchell,
            CubicCatmullRom,
            LinearWithLinearMipmap,
            LinearWithNearestMipmap,
            LinearNoMipmap,
            NearestWithLinearMipmap,
            NearestWithNearestMipmap,
            NearestNoMipmap,
            Anisotropic4x,
            Anisotropic8x,
            Anisotropic16x
        }

        [GlobalSetup]
        public void GlobalSetup()
        {
            _outputImageInfo = new SKImageInfo(_width, _height, SKColorType.Rgb888x, SKAlphaType.Opaque);
            _pinnedBufferPool = new PinnedMemoryBufferPool(_outputImageInfo);

            _image = SKBitmap.Decode(_testImage);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _image?.Dispose();
        }

        [Params(SamplingProfile.CubicMitchell,
            SamplingProfile.CubicCatmullRom,
            SamplingProfile.LinearWithLinearMipmap,
            SamplingProfile.LinearWithNearestMipmap,
            SamplingProfile.LinearNoMipmap,
            SamplingProfile.NearestWithLinearMipmap,
            SamplingProfile.NearestWithNearestMipmap,
            SamplingProfile.NearestNoMipmap,
            SamplingProfile.Anisotropic4x,
            SamplingProfile.Anisotropic8x,
            SamplingProfile.Anisotropic16x
            )]
        public SamplingProfile Profile { get; set; }

        private SKSamplingOptions SamplingOptions
        {
            get
            {
                return Profile switch
                {
                    /// <summary>
                    /// Cubic Mitchell resampling.
                    /// - <b>Quality:</b> High (smooth results, reduced aliasing).
                    /// - <b>Performance:</b> Moderate.
                    /// - <b>Best Use Case:</b> When high-quality resampling is needed.
                    /// </summary>
                    SamplingProfile.CubicMitchell => new SKSamplingOptions(SKCubicResampler.Mitchell),

                    /// <summary>
                    /// Cubic Catmull-Rom resampling.
                    /// - <b>Quality:</b> Very sharp (preserves details well but can introduce ringing artifacts).
                    /// - <b>Performance:</b> Moderate.
                    /// - <b>Best Use Case:</b> When sharpness is more important than smoothness.
                    /// </summary>
                    SamplingProfile.CubicCatmullRom => new SKSamplingOptions(SKCubicResampler.CatmullRom),

                    /// <summary>
                    /// Linear filtering with linear mipmap interpolation.
                    /// - <b>Quality:</b> Balanced (smooth with slight blur).
                    /// - <b>Performance:</b> Good.
                    /// - <b>Best Use Case:</b> When scaling down with a balance of speed and quality.
                    /// </summary>
                    SamplingProfile.LinearWithLinearMipmap => new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear),

                    /// <summary>
                    /// Linear filtering with nearest mipmap interpolation.
                    /// - <b>Quality:</b> Slightly sharper than linear mipmap.
                    /// - <b>Performance:</b> Good.
                    /// - <b>Best Use Case:</b> When performance is slightly prioritized over smoothness.
                    /// </summary>
                    SamplingProfile.LinearWithNearestMipmap => new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Nearest),

                    /// <summary>
                    /// Linear filtering without mipmap (default in YoloDotNet).
                    /// - <b>Quality:</b> Decent (default option, slight blur).
                    /// - <b>Performance:</b> Fast.
                    /// - <b>Best Use Case:</b> General-purpose resizing with a balance of speed and quality.
                    /// </summary>
                    SamplingProfile.LinearNoMipmap => new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None),

                    /// <summary>
                    /// Nearest filtering with linear mipmap interpolation.
                    /// - <b>Quality:</b> Lower (visible pixelation).
                    /// - <b>Performance:</b> Very fast.
                    /// - <b>Best Use Case:</b> When maximum speed is needed but mipmaps help reduce aliasing.
                    /// </summary>
                    SamplingProfile.NearestWithLinearMipmap => new SKSamplingOptions(SKFilterMode.Nearest, SKMipmapMode.Linear),

                    /// <summary>
                    /// Nearest filtering with nearest mipmap interpolation.
                    /// - <b>Quality:</b> Low (pixelated, harsh transitions).
                    /// - <b>Performance:</b> Very fast.
                    /// - <b>Best Use Case:</b> When raw performance is the top priority.
                    /// </summary>
                    SamplingProfile.NearestWithNearestMipmap => new SKSamplingOptions(SKFilterMode.Nearest, SKMipmapMode.Nearest),

                    /// <summary>
                    /// Nearest filtering without mipmap.
                    /// - <b>Quality:</b> Lowest (hard edges, pixelation).
                    /// - <b>Performance:</b> Fastest.
                    /// - <b>Best Use Case:</b> When performance is critical, and quality is not a concern.
                    /// </summary>
                    SamplingProfile.NearestNoMipmap => new SKSamplingOptions(SKFilterMode.Nearest, SKMipmapMode.None),

                    /// <summary>
                    /// 4x Anisotropic filtering.
                    /// - <b>Quality:</b> Good (reduces blurring on angled textures).
                    /// - <b>Performance:</b> Moderate.
                    /// - <b>Best Use Case:</b> When dealing with textures viewed at an angle.
                    /// </summary>
                    SamplingProfile.Anisotropic4x => new SKSamplingOptions(4),

                    /// <summary>
                    /// 8x Anisotropic filtering.
                    /// - <b>Quality:</b> Very good (better sharpness on textures at an angle).
                    /// - <b>Performance:</b> Lower than 4x but still efficient.
                    /// - <b>Best Use Case:</b> When dealing with angled textures and need better detail.
                    /// </summary>
                    SamplingProfile.Anisotropic8x => new SKSamplingOptions(8),

                    /// <summary>
                    /// 16x Anisotropic filtering.
                    /// - <b>Quality:</b> Best (preserves details on angled textures).
                    /// - <b>Performance:</b> Slowest among anisotropic options.
                    /// - <b>Best Use Case:</b> When the highest texture quality is required, especially for angled surfaces.
                    /// </summary>
                    SamplingProfile.Anisotropic16x => new SKSamplingOptions(16),
                    _ => throw new ArgumentOutOfRangeException()
                };
            }
        }

        [Benchmark]
        public void Resize()
        {
            var pinnedBuffer = _pinnedBufferPool.Rent();

            try
            {
                _ = _image.ResizeImageProportional<SKBitmap>(SamplingOptions, pinnedBuffer);
            }
            finally
            {
                _pinnedBufferPool.Return(pinnedBuffer);
            }
        }
    }
}
