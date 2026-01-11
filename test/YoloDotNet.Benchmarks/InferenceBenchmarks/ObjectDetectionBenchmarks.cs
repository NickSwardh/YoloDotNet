// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.InferenceBenchmarks
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class ObjectDetectionBenchmarks
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);
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
            _image?.Dispose();
            _yolo.Dispose();
        }

        /// <summary>
        /// Specifies which YoloDotNet configuration to benchmark. Each entry represents a distinct combination 
        /// of YOLO version, model type, and execution backend (e.g., CPU, CUDA, or TensorRT with FP32/FP16/INT8 precision).
        ///
        /// ⚠️ For TensorRT benchmarks (e.g., TRT32, TRT16, TRT8), ensure that <see cref="TensorRtConfig"/> is properly configured 
        /// to define the directory path used for storing and loading TensorRT engine cache files. 
        /// Engine caching significantly improves performance by avoiding runtime engine re-compilation.
        ///
        /// This parameter will be used by BenchmarkDotNet to generate individual benchmark cases 
        /// for each specified configuration.
        /// </summary>
        [Params(
            YoloType.V5u_Obj_CPU,
            YoloType.V5u_Obj_GPU,
            //YoloType.V5u_Obj_TRT32,
            //YoloType.V5u_Obj_TRT16,
            //YoloType.V5u_Obj_TRT8,

            YoloType.V8_Obj_CPU,
            YoloType.V8_Obj_GPU,
            //YoloType.V8_Obj_TRT32,
            //YoloType.V8_Obj_TRT16,
            //YoloType.V8_Obj_TRT8,

            YoloType.V9_Obj_CPU,
            YoloType.V9_Obj_GPU,
            //YoloType.V9_Obj_TRT32,
            //YoloType.V9_Obj_TRT16,
            //YoloType.V9_Obj_TRT8,

            YoloType.V10_Obj_CPU,
            YoloType.V10_Obj_GPU,
            //YoloType.V10_Obj_TRT32,
            //YoloType.V10_Obj_TRT16,
            //YoloType.V10_Obj_TRT8,

            YoloType.V11_Obj_CPU,
            YoloType.V11_Obj_GPU,
            //YoloType.V11_Obj_TRT32,
            //YoloType.V11_Obj_TRT16,
            //YoloType.V11_Obj_TRT8,

            YoloType.V12_Obj_CPU,
            YoloType.V12_Obj_GPU
            //YoloType.V12_Obj_TRT32,
            //YoloType.V12_Obj_TRT16,
            //YoloType.V12_Obj_TRT8
            )]
        public YoloType YoloParam { get; set; }

        [Benchmark]
        public void ObjectDetection()
            => _yolo.RunObjectDetection(_image);
    }
}
