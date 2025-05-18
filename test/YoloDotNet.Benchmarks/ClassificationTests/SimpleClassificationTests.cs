using Microsoft.VSDiagnostics;

namespace YoloDotNet.Benchmarks.ClassificationTests
{
    //[CPUUsageDiagnoser]
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

        private SKImage _skImage;
        private SKBitmap _skBitmap;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            var options = new YoloOptions();

            _skBitmap = SKBitmap.Decode(_testImage);
            _skImage = SKImage.FromEncodedData(_testImage);

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
            _gpuYolov8?.Dispose();

            _cpuYolov11?.Dispose();
            _gpuYolov11?.Dispose();

            _skBitmap?.Dispose();
            _skImage?.Dispose();
        }

        // Yolov8 CPU
        [Benchmark]
        public void RunSimpleClassificationYolov8CpuOnSKImage()
        {
            _ = _cpuYolov8.RunClassification(_skImage);
        }

        [Benchmark]
        public void RunSimpleClassificationYolov8CpuOnSKBitmap()
        {
            _ = _cpuYolov8.RunClassification(_skBitmap);
        }


        // Yolov8 GPU
        [Benchmark]
        public void RunSimpleClassificationYolov8GpuOnSKImage()
        {
            _ = _gpuYolov8.RunClassification(_skImage);
        }

        [Benchmark]
        public void RunSimpleClassificationYolov8GpuOnSKBitmap()
        {
            _ = _gpuYolov8.RunClassification(_skBitmap);
        }


        // Yolov11 CPU
        [Benchmark]
        public void RunSimpleClassificationYolov11CpuOnSKImage()
        {
            _ = _cpuYolov11.RunClassification(_skImage);
        }

        [Benchmark]
        public void RunSimpleClassificationYolov11CpuOnSKBitmap()
        {
            _ = _cpuYolov11.RunClassification(_skBitmap);
        }


        // Yolov11 GPU
        [Benchmark]
        public void RunSimpleClassificationYolov11GpuOnSKImage()
        {
            _ = _gpuYolov11.RunClassification(_skImage);
        }

        [Benchmark]
        public void RunSimpleClassificationYolov11GpuOnSKBitmap()
        {
            _ = _gpuYolov11.RunClassification(_skBitmap);
        }

        #endregion Methods
    }
}
