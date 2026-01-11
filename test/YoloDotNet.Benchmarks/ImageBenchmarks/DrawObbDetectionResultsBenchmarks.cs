// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ImageBenchmarks
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class DrawObbDetectionResultsBenchmarks
    {
        private static readonly string _testImage = SharedConfig.GetTestImage(ImageType.Island);

        private Yolo _yolo;
        private SKBitmap _image;
        private List<Models.OBBDetection> _detectionResults;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKBitmap.Decode(_testImage);

            _yolo = YoloCreator.Create(YoloType.V11_Obb_CPU);
            _detectionResults = _yolo.RunObbDetection(_image);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _yolo?.Dispose();
            _image?.Dispose();
        }

        [Benchmark]
        public void DrawObbDetectionResults()
            => _image.Draw(_detectionResults);

    }
}
