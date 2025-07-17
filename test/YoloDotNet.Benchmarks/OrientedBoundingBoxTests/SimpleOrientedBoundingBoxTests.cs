﻿// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.OrientedBoundingBoxTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class SimpleOrientedBoundingBoxTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Island);
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
            YoloType.V8_Obb_CPU,
            YoloType.V8_Obb_GPU,
            YoloType.V8_Obb_TRT32,
            YoloType.V8_Obb_TRT16,
            YoloType.V8_Obb_TRT8,

            YoloType.V11_Obb_CPU,
            YoloType.V11_Obb_GPU,
            YoloType.V11_Obb_TRT32,
            YoloType.V11_Obb_TRT16,
            YoloType.V11_Obb_TRT8
            )]
        public YoloType YoloParam { get; set; }

        [Benchmark]
        public List<OBBDetection> ObbDetection()
            => _yolo.RunObbDetection(_image);
    }
}
