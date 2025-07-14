// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class SegmentationImageDrawTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.People);

        private List<Segmentation> _segmentations;
        private SKBitmap _image;
        private Yolo _yolo;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _yolo = YoloCreator.CreateYolo(YoloType.V8_Seg_CPU);
            _image = SKBitmap.Decode(_testImage);
            _segmentations = _yolo.RunSegmentation(_image);
        }

        [GlobalCleanup]
        public void CleanUp()
        {
            _yolo?.Dispose();
            _image?.Dispose();
        }

        [Benchmark]
        public void DrawSegmentation()
            => _image.Draw(_segmentations);
    }
}
