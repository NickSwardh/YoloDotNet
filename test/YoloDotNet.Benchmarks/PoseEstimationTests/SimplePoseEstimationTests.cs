﻿namespace YoloDotNet.Benchmarks.PoseEstimationTests
{
    [MemoryDiagnoser]
    public class SimplePoseEstimationTests
    {
        #region Fields

        private readonly string _model8 = SharedConfig.GetTestModelV8(ModelType.PoseEstimation);
        private readonly string _model11 = SharedConfig.GetTestModelV11(ModelType.PoseEstimation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

        private Yolo _gpuYolov8;
        private Yolo _cpuYolov8;

        private Yolo _gpuYolov11;
        private Yolo _cpuYolov11;

        private SKImage _image;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKImage.FromEncodedData(_testImage);

            var options = new YoloOptions
            {
                ModelType = ModelType.PoseEstimation,
            };

            // Yolov8
            options.OnnxModel = _model8;

            options.HwAccelerator = HwAcceleratorType.None;
            _cpuYolov8 = new Yolo(options);

            options.HwAccelerator = HwAcceleratorType.Cuda;
            _gpuYolov8 = new Yolo(options);

            // Yolov11
            options.OnnxModel = _model11;

            options.HwAccelerator = HwAcceleratorType.None;
            _cpuYolov11 = new Yolo(options);

            options.HwAccelerator = HwAcceleratorType.Cuda;
            _gpuYolov11 = new Yolo(options);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cpuYolov8?.Dispose();
            _cpuYolov11?.Dispose();
            _gpuYolov8?.Dispose();
            _gpuYolov11?.Dispose();
            _image?.Dispose();
        }

        // Yolov8
        [Benchmark]
        public List<PoseEstimation> RunSimplePoseEstimationYolov8Cpu()
        {
            return _cpuYolov8.RunPoseEstimation(_image);
        }

        [Benchmark]
        public List<PoseEstimation> RunSimplePoseEstimationYolov8Gpu()
        {
            return _gpuYolov8.RunPoseEstimation(_image);
        }

        // Yolov11
        [Benchmark]
        public List<PoseEstimation> RunSimplePoseEstimationYolov11Cpu()
        {
            return _cpuYolov11.RunPoseEstimation(_image);
        }

        [Benchmark]
        public List<PoseEstimation> RunSimplePoseEstimationYolov11Gpu()
        {
            return _gpuYolov11.RunPoseEstimation(_image);
        }

        #endregion Methods
    }
}
