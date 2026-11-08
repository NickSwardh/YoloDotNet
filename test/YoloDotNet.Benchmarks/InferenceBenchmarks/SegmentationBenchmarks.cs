// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.InferenceBenchmarks
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class SegmentationBenchmarks
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Segmentation);
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
            YoloType.V8_Seg_GPUGraph,
            YoloType.V8_Seg_TRT32,
            YoloType.V8_Seg_TRT16,
            YoloType.V8_Seg_TRT32Graph,
            YoloType.V8_Seg_TRT16Graph,

            YoloType.V11_Seg_CPU,
            YoloType.V11_Seg_GPU,
            YoloType.V11_Seg_GPUGraph,
            YoloType.V11_Seg_TRT32,
            YoloType.V11_Seg_TRT16,
            YoloType.V11_Seg_TRT32Graph,
            YoloType.V11_Seg_TRT16Graph,

            YoloType.V26_Seg_CPU,
            YoloType.V26_Seg_GPU,
            YoloType.V26_Seg_GPUGraph,
            YoloType.V26_Seg_TRT32,
            YoloType.V26_Seg_TRT16,
            YoloType.V26_Seg_TRT32Graph,
            YoloType.V26_Seg_TRT16Graph
            )]
        public YoloType YoloParam { get; set; }

        [Benchmark]
        public List<Segmentation> Segmentation()
            => _yolo.RunSegmentation(_image);
    }
}
