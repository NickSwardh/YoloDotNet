﻿// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ObjectDetectionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class SimpleObjectDetectionTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);
        private SKBitmap _image;
        private Yolo _yolo;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKBitmap.Decode(_testImage);
            _yolo = YoloCreator.CreateYolo(YoloParam);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _image?.Dispose();
            _yolo.Dispose();
        }

        [Params(YoloType.V5u_Obj_CPU,
            YoloType.V5u_Obj_GPU,
            YoloType.V8_Obj_CPU,
            YoloType.V8_Obj_GPU,
            YoloType.V9_Obj_CPU,
            YoloType.V9_Obj_GPU,
            YoloType.V10_Obj_CPU,
            YoloType.V10_Obj_GPU,
            YoloType.V11_Obj_CPU,
            YoloType.V11_Obj_GPU,
            YoloType.V12_Obj_CPU,
            YoloType.V12_Obj_GPU
            )]
        public YoloType YoloParam { get; set; }

        [Benchmark]
        public void ObjectDetection()
            => _yolo.RunObjectDetection(_image);
    }
}
