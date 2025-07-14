// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class OrientedBoundingBoxImageDrawTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Island);

        private Yolo _yolo;
        private SKBitmap _skBitmap;
        private List<OBBDetection> _obbDetections;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _yolo = YoloCreator.CreateYolo(YoloType.V8_Obb_CPU);
            _skBitmap = SKBitmap.Decode(_testImage);

            // We just need one result to use for drawing.
            _obbDetections = _yolo.RunObbDetection(_skBitmap);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _yolo?.Dispose();
            _skBitmap?.Dispose();
        }

        [Benchmark]
        public void DrawObb()
            => _skBitmap.Draw(_obbDetections);
    }
}
