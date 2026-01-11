// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.InferenceBenchmarks
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class SegmentationBenchmarks
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.People);
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

        // Segmentation with TensorRT INT8 precision not supported.
        [Params(
            YoloType.V8_Seg_CPU,
            YoloType.V8_Seg_GPU,
            //YoloType.V8_Seg_TRT32,
            //YoloType.V8_Seg_TRT16,
            YoloType.V11_Seg_CPU,
            YoloType.V11_Seg_GPU
            //YoloType.V11_Seg_TRT32,
            //YoloType.V11_Seg_TRT16
            )]
        public YoloType YoloParam { get; set; }

        [Benchmark]
        public List<Segmentation> Segmentation()
            => _yolo.RunSegmentation(_image);
    }
}
