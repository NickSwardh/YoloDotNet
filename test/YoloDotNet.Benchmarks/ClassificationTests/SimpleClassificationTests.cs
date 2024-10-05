namespace YoloDotNet.Benchmarks.ClassificationTests
{
    [MemoryDiagnoser]
    public class SimpleClassificationTests
    {
        #region Fields

        private static readonly string _model8 = SharedConfig.GetTestModelV8(ModelType.Classification);
        private static readonly string _model11 = SharedConfig.GetTestModelV11(ModelType.Classification);
        private static readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

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
            var options = new YoloOptions
            {
                ModelType = ModelType.Classification
            };

            _image = SKImage.FromEncodedData(_testImage);

            // Yolov8
            options.OnnxModel = _model8;
            options.ModelVersion = ModelVersion.V8;

            options.Cuda = false;
            _cpuYolov8 = new Yolo(options);

            options.Cuda = true;
            _gpuYolov8 = new Yolo(options);

            // Yolov11
            options.OnnxModel = _model11;
            options.ModelVersion = ModelVersion.V11;

            options.Cuda = false;
            _cpuYolov11 = new Yolo(options);

            options.Cuda = true;
            _gpuYolov11 = new Yolo(options);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cpuYolov8?.Dispose();
            _cpuYolov11?.Dispose();
            _gpuYolov8?.Dispose();
            _gpuYolov11?.Dispose();
            _image.Dispose();
        }

        // Yolov8
        [Benchmark]
        public void RunSimpleClassificationYolov8Cpu()
        {
            _ = _cpuYolov8.RunClassification(_image);
        }

        [Benchmark]
        public void RunSimpleClassificationYolov8Gpu()
        {
            _ = _gpuYolov8.RunClassification(_image);
        }

        // Yolov11
        [Benchmark]
        public void RunSimpleClassificationYolov11Cpu()
        {
            _ = _cpuYolov11.RunClassification(_image);
        }

        [Benchmark]
        public void RunSimpleClassificationYolov111Gpu()
        {
            _ = _gpuYolov11.RunClassification(_image);
        }

        #endregion Methods
    }
}
