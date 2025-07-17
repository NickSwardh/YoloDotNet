// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.PoseEstimationTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class SimplePoseEstimationTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

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
            _image?.Dispose();
            _yolo?.Dispose();
        }

        [Params(YoloType.V8_Pos_CPU,
            YoloType.V8_Pos_GPU,
            YoloType.V8_Pos_TRT32,
            YoloType.V8_Pos_TRT16,
            YoloType.V8_Pos_TRT8,

            YoloType.V11_Pos_CPU,
            YoloType.V11_Pos_GPU,
            YoloType.V11_Pos_TRT32,
            YoloType.V11_Pos_TRT16,
            YoloType.V11_Pos_TRT8
            )]
        public YoloType YoloParam { get; set; }

        [Benchmark]
        public List<PoseEstimation> PoseEstimation()
            => _yolo.RunPoseEstimation(_image);
    }
}
