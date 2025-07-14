// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class ObjectDetectionImageDrawTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);

        private Yolo _yolo;
        private SKBitmap _image;
        private List<ObjectDetection> _objectDetections;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _yolo = YoloCreator.CreateYolo(YoloType.V8_Obj_CPU);
            _image = SKBitmap.Decode(_testImage);

            _objectDetections = _yolo.RunObjectDetection(_image);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _yolo?.Dispose();
            _image?.Dispose();
        }

        [Benchmark]
        public void DrawObjectDetection()
            => _image.Draw(_objectDetections);
    }
}
