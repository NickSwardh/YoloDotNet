// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.SegmentationTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class SimpleSegmentationTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.People);
        private SKBitmap _image;
        private Yolo _yolo;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKBitmap.Decode(_testImage);
            _yolo = YoloCreator.Create(YoloParam);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _yolo?.Dispose();
            _image?.Dispose();
        }

        [Params(
            YoloType.V8_Seg_CPU,
            YoloType.V8_Seg_GPU,
            YoloType.V11_Seg_CPU,
            YoloType.V11_Seg_GPU
            )]
        public YoloType YoloParam { get; set; }

        [Benchmark]
        public List<Segmentation> Segmentation()
            => _yolo.RunSegmentation(_image);
    }
}
