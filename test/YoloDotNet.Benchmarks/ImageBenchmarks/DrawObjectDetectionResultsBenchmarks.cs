// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ImageBenchmarks
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class DrawObjectDetectionResultsBenchmarks
    {
        private static readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);

        private Yolo _yolo;
        private SKBitmap _image;
        private List<ObjectDetection> _detectionResults;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKBitmap.Decode(_testImage);

            _yolo = YoloCreator.Create(YoloType.V11_Obj_CPU);
            _detectionResults = _yolo.RunObjectDetection(_image);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _yolo?.Dispose();
            _image?.Dispose();
        }

        [Benchmark]
        public void DrawObjectDetectionResults()
            => _image.Draw(_detectionResults);

    }
}
