namespace YoloDotNet.Benchmarks.ObjectDetectionTests
{
    [MemoryDiagnoser]
    public class SimpleObjectDetectionTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);
        private readonly string _modelV8 = SharedConfig.GetTestModelV8(ModelType.ObjectDetection);
        private readonly string _modelV9 = SharedConfig.GetTestModelV9(ModelType.ObjectDetection);
        private readonly string _modelV10 = SharedConfig.GetTestModelV10(ModelType.ObjectDetection);
        private readonly string _modelV11 = SharedConfig.GetTestModelV11(ModelType.ObjectDetection);

        private Yolo _cpuYolov8;
        private Yolo _gpuYolov8;

        private Yolo _cpuYolov9;
        private Yolo _gpuYolov9;

        private Yolo _cpuYolov10;
        private Yolo _gpuYolov10;

        private Yolo _cpuYolov11;
        private Yolo _gpuYolov11;

        private SKImage _image;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKImage.FromEncodedData(_testImage);

            var options = new YoloOptions
            {
                ModelType = ModelType.ObjectDetection,
                HwAccelerator = HwAcceleratorType.None
            };

            // Yolov8
            options.OnnxModel = _modelV8;
            _cpuYolov8 = new Yolo(options);

            options.HwAccelerator = HwAcceleratorType.Cuda;
            _gpuYolov8 = new Yolo(options);

            // Yolov9
            options.OnnxModel = _modelV9;
            options.HwAccelerator = HwAcceleratorType.None;

            _cpuYolov9 = new Yolo(options);

            options.HwAccelerator = HwAcceleratorType.Cuda;
            _gpuYolov9 = new Yolo(options);

            // Yolov10
            options.OnnxModel = _modelV10;
            options.HwAccelerator = HwAcceleratorType.None;

            _cpuYolov10 = new Yolo(options);

            options.HwAccelerator = HwAcceleratorType.Cuda;
            _gpuYolov10 = new Yolo(options);

            // Yolov11
            options.OnnxModel = _modelV11;
            options.HwAccelerator = HwAcceleratorType.None;

            _cpuYolov11 = new Yolo(options);

            options.HwAccelerator = HwAcceleratorType.Cuda;
            _gpuYolov11 = new Yolo(options);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _image.Dispose();
            _cpuYolov8?.Dispose();
            _gpuYolov8?.Dispose();
            _cpuYolov9?.Dispose();
            _gpuYolov9?.Dispose();
            _cpuYolov10?.Dispose();
            _gpuYolov10?.Dispose();
            _cpuYolov11?.Dispose();
            _gpuYolov11?.Dispose();
        }

        [Benchmark]
        public void ObjectDetectionYolov8Cpu()
        {
            _ = _cpuYolov8.RunObjectDetection(_image);
        }

        [Benchmark]
        public void ObjectDetectionYolov8Gpu()
        {
            _ = _gpuYolov8.RunObjectDetection(_image);
        }

        [Benchmark]
        public void ObjectDetectionYolov9Cpu()
        {
            _ = _cpuYolov9.RunObjectDetection(_image);
        }

        [Benchmark]
        public void ObjectDetectionYolov9Gpu()
        {
            _ = _gpuYolov9.RunObjectDetection(_image);
        }

        [Benchmark]
        public void ObjectDetectionYolov10Cpu()
        {
            _ = _cpuYolov10.RunObjectDetection(_image);
        }

        [Benchmark]
        public void ObjectDetectionYolov10Gpu()
        {
            _ = _gpuYolov10.RunObjectDetection(_image);
        }

        [Benchmark]
        public void ObjectDetectionYolov11Cpu()
        {
            _ = _cpuYolov11.RunObjectDetection(_image);
        }

        [Benchmark]
        public void ObjectDetectionYolov11Gpu()
        {
            _ = _gpuYolov11.RunObjectDetection(_image);
        }
    }
}
