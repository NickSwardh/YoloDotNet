// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.ImageBenchmarks
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class DrawPoseEstimationResultsBenchmarks
    {
        private static readonly string _testImage = SharedConfig.GetTestImage(ImageType.People);

        private Yolo _yolo;
        private SKBitmap _image;
        private List<Models.PoseEstimation> _detectionResults;
        private PoseDrawingOptions _drawingOptions;

        public DrawPoseEstimationResultsBenchmarks()
        {
            _drawingOptions = new PoseDrawingOptions
            {
                KeyPointMarkers = CustomKeyPointColorMap.KeyPoints
            };
        }

        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKBitmap.Decode(_testImage);

            _yolo = YoloCreator.Create(YoloType.V11_Pos_CPU);
            _detectionResults = _yolo.RunPoseEstimation(_image);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _yolo?.Dispose();
            _image?.Dispose();
        }

        [Benchmark]
        public void DrawPoseEstimationResults()
            => _image.Draw(_detectionResults, _drawingOptions);

    }
}
