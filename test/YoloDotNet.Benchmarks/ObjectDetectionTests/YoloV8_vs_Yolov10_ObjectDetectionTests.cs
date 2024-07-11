namespace YoloDotNet.Benchmarks.ObjectDetectionTests
{
    [MemoryDiagnoser]
    public class YoloV8_vs_Yolov10_ObjectDetectionTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);
        private readonly string _modelV8 = SharedConfig.GetTestModel(ModelType.ObjectDetection);
        private readonly string _modelv10 = SharedConfig.GetTestModelV10(ModelType.ObjectDetection);

        private Yolo _cpuYolov8;

        private Yolo _gpuYolov8;

        private Yolo _cpuYolov10;

        private Yolo _gpuYolov10;

        private SKImage _image;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKImage.FromEncodedData(_testImage);

            var options = new YoloOptions
            {
                OnnxModel = _modelV8,
                ModelType = ModelType.ObjectDetection,
                Cuda = false
            };

            // Yolov8
            _cpuYolov8 = new Yolo(options);

            options.Cuda = true;
            _gpuYolov8 = new Yolo(options);

            // Yolov10
            options.OnnxModel = _modelv10;
            options.Cuda = false;

            _cpuYolov10 = new Yolo(options);

            options.Cuda = true;
            _gpuYolov10 = new Yolo(options);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _image.Dispose();
            _cpuYolov8.Dispose();
            _gpuYolov8.Dispose();
            _cpuYolov10.Dispose();
            _gpuYolov10.Dispose();
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
        public void ObjectDetectionYolov10Cpu()
        {
            _ = _cpuYolov10.RunObjectDetection(_image);
        }

        [Benchmark]
        public void ObjectDetectionYolov10Gpu()
        {
            _ = _gpuYolov10.RunObjectDetection(_image);
        }
    }
}
