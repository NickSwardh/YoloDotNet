// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ImageBenchmarks
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class ImageResizeBenchmarks
    {
        private static readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);

        private PinnedMemoryBuffer _pinnedMemoryBuffer;
        private SKBitmap _image;
        private readonly SKSamplingOptions _samplingOptions;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKBitmap.Decode(_testImage);
            _pinnedMemoryBuffer = new PinnedMemoryBuffer(new SKImageInfo(640,640, SKColorType.Rgb888x, SKAlphaType.Opaque));
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _image?.Dispose();
            _pinnedMemoryBuffer?.Dispose();
        }


        [Benchmark]
        public void ResizeProportional()
            => _image.ResizeImageProportional(_samplingOptions, _pinnedMemoryBuffer);

        [Benchmark]
        public void ResizeStretched()
            => _image.ResizeImageStretched(_samplingOptions, _pinnedMemoryBuffer);
    }
}
