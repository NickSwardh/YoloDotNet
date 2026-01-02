// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.PoseEstimation
{
    namespace YoloDotNet.Benchmarks.ClassificationTests
    {
        //[CPUUsageDiagnoser]
        [MemoryDiagnoser]
        public class PoseEstimationBenchmarks
        {
            private static readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

            private Yolo _yolo;
            private SKBitmap _image;

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

            [Params(YoloType.V8_Cls_CPU,
                YoloType.V8_Cls_GPU,
                //YoloType.V8_Cls_TRT32,
                //YoloType.V8_Cls_TRT16,
                //YoloType.V8_Cls_TRT8,

                YoloType.V11_Cls_CPU,
                YoloType.V11_Cls_GPU
                //YoloType.V11_Cls_TRT32,
                //YoloType.V11_Cls_TRT16,
                //YoloType.V11_Cls_TRT8
                )]
            public YoloType YoloParam { get; set; }

            [Benchmark]
            public void Classification()
                => _yolo.RunClassification(_image);
        }
    }
}
