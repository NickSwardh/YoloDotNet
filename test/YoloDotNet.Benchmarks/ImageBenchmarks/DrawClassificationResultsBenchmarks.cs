// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ImageBenchmarks
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class DrawClassificationResultsBenchmarks
    {
        private static readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private Yolo _yolo;
        private SKBitmap _image;
        private List<Models.Classification> _detectionResults;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKBitmap.Decode(_testImage);

            _yolo = YoloCreator.Create(YoloType.V11_Cls_CPU);
            _detectionResults = _yolo.RunClassification(_image);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _yolo?.Dispose();
            _image?.Dispose();
        }

        [Benchmark]
        public void DrawClassificationResults()
            => _image.Draw(_detectionResults);

    }
}
