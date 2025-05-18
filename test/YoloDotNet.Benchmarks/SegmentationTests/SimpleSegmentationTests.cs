namespace YoloDotNet.Benchmarks.SegmentationTests
{
    [MemoryDiagnoser]
    public class SimpleSegmentationTests
    {
        #region Fields

        private readonly string _model8 = SharedConfig.GetTestModelV8(ModelType.Segmentation);
        private readonly string _model11 = SharedConfig.GetTestModelV11(ModelType.Segmentation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.People);

        private Yolo _gpuYolov8;
        private Yolo _cpuYolov8;

        private Yolo _gpuYolov11;
        private Yolo _cpuYolov11;

        private SKBitmap _image;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            var options = new YoloOptions();

            _image = SKBitmap.Decode(_testImage);

            // Yolov8
            options.OnnxModel = _model8;

            options.Cuda = false;
            _cpuYolov8 = new Yolo(options);

            options.Cuda = true;
            _gpuYolov8 = new Yolo(options);

            // Yolov11
            options.OnnxModel = _model11;

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
        public List<Segmentation> RunSimpleSegmentationYolov8Cpu()
        {
            return _cpuYolov8.RunSegmentation(_image);
        }

        [Benchmark]
        public List<Segmentation> RunSimpleSegmentationYolov8Gpu()
        {
            return _gpuYolov8.RunSegmentation(_image);
        }

        // Yolov11
        [Benchmark]
        public List<Segmentation> RunSimpleSegmentationYolov11Cpu()
        {
            return _cpuYolov11.RunSegmentation(_image);
        }

        [Benchmark]
        public List<Segmentation> RunSimpleSegmentationYolov11Gpu()
        {
            return _gpuYolov11.RunSegmentation(_image);
        }

        #endregion Methods
    }
}
