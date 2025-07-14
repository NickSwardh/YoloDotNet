// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ClassificationTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class SimpleClassificationTests
    {
        private static readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private Yolo _yolo;
        private SKBitmap _image;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKBitmap.Decode(_testImage);

            _yolo = YoloCreator.CreateYolo(YoloParam);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _yolo?.Dispose();
            _image?.Dispose();
        }

        [Params(YoloType.V8_Cls_CPU,
            YoloType.V8_Cls_GPU,
            YoloType.V11_Cls_CPU,
            YoloType.V11_Cls_GPU
            )]
        public YoloType YoloParam { get; set; }

        [Benchmark]
        public void Classification()
            => _yolo.RunClassification(_image);
    }
}
