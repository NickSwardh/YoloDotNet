// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class PoseEstimationImageDrawTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

        private Yolo _yolo;
        private SKBitmap _image;
        private List<PoseEstimation> _poseEstimations;
        private PoseDrawingOptions _poseDrawingOptions;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _yolo = YoloCreator.Create(YoloType.V8_Pos_CPU);
            _image = SKBitmap.Decode(_testImage);
            _poseEstimations = _yolo.RunPoseEstimation(_image);

            _poseDrawingOptions = new PoseDrawingOptions
            {
                KeyPointMarkers = CustomKeyPointColorMap.KeyPoints
            };
        }

        [GlobalCleanup]
        public void CleanUp()
        {
            _yolo?.Dispose();
            _image?.Dispose();
        }

        [Benchmark]
        public void DrawPoseEstimation()
            => _image.Draw(_poseEstimations, _poseDrawingOptions);
    }
}
