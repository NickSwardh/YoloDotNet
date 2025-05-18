namespace YoloDotNet.Benchmarks.ObjectDetectionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class SimpleObjectDetectionTests
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);
        private readonly string _modelV5U = SharedConfig.GetTestModelV5U(ModelType.ObjectDetection);
        private readonly string _modelV8 = SharedConfig.GetTestModelV8(ModelType.ObjectDetection);
        private readonly string _modelV9 = SharedConfig.GetTestModelV9(ModelType.ObjectDetection);
        private readonly string _modelV10 = SharedConfig.GetTestModelV10(ModelType.ObjectDetection);
        private readonly string _modelV11 = SharedConfig.GetTestModelV11(ModelType.ObjectDetection);
        private readonly string _modelV12 = SharedConfig.GetTestModelV12(ModelType.ObjectDetection);

        private Yolo _cpuYolov5u;
        private Yolo _gpuYolov5u;

        private Yolo _cpuYolov8;
        private Yolo _gpuYolov8;

        private Yolo _cpuYolov9;
        private Yolo _gpuYolov9;

        private Yolo _cpuYolov10;
        private Yolo _gpuYolov10;

        private Yolo _cpuYolov11;
        private Yolo _gpuYolov11;

        private Yolo _cpuYolov12;
        private Yolo _gpuYolov12;

        private SKBitmap _skBitmap;
        private SKImage _skImage;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _skBitmap = SKBitmap.Decode(_testImage);
            _skImage = SKImage.FromEncodedData(_testImage);

            var options = new YoloOptions
            {
                Cuda = false
            };

            // Yolov5U
            options.OnnxModel = _modelV5U;
            options.Cuda = false;

            _cpuYolov5u = new Yolo(options);

            options.Cuda = true;
            _gpuYolov5u = new Yolo(options);

            // Yolov8
            options.OnnxModel = _modelV8;
            options.Cuda = false;

            _cpuYolov8 = new Yolo(options);

            options.Cuda = true;
            _gpuYolov8 = new Yolo(options);

            // Yolov9
            options.OnnxModel = _modelV9;
            options.Cuda = false;

            _cpuYolov9 = new Yolo(options);

            options.Cuda = true;
            _gpuYolov9 = new Yolo(options);

            // Yolov10
            options.OnnxModel = _modelV10;
            options.Cuda = false;

            _cpuYolov10 = new Yolo(options);

            options.Cuda = true;
            _gpuYolov10 = new Yolo(options);

            // Yolov11
            options.OnnxModel = _modelV11;
            options.Cuda = false;

            _cpuYolov11 = new Yolo(options);

            options.Cuda = true;
            _gpuYolov11 = new Yolo(options);

            // Yolov12
            options.OnnxModel = _modelV12;
            options.Cuda = false;

            _cpuYolov12 = new Yolo(options);

            options.Cuda = true;
            _gpuYolov12 = new Yolo(options);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cpuYolov5u?.Dispose();
            _gpuYolov5u?.Dispose();
            _cpuYolov8?.Dispose();
            _gpuYolov8?.Dispose();
            _cpuYolov9?.Dispose();
            _gpuYolov9?.Dispose();
            _cpuYolov10?.Dispose();
            _gpuYolov10?.Dispose();
            _cpuYolov11?.Dispose();
            _gpuYolov11?.Dispose();
            _cpuYolov12?.Dispose();
            _gpuYolov12?.Dispose();

            _skImage?.Dispose();
            _skBitmap?.Dispose();
        }

        [Benchmark]
        public void ObjectDetectionYolov5uCpuSKImage()
        {
            _ = _cpuYolov5u.RunObjectDetection(_skImage);
        }

        [Benchmark]
        public void ObjectDetectionYolov5uCpuSKBitmap()
        {
            _ = _cpuYolov5u.RunObjectDetection(_skBitmap);
        }



        [Benchmark]
        public void ObjectDetectionYolov8CpuSKImage()
        {
            _ = _cpuYolov8.RunObjectDetection(_skImage);
        }

        [Benchmark]
        public void ObjectDetectionYolov8CpuSKBitmap()
        {
            _ = _cpuYolov8.RunObjectDetection(_skBitmap);
        }




        [Benchmark]
        public void ObjectDetectionYolov8GpuSKImage()
        {
            _ = _gpuYolov8.RunObjectDetection(_skImage);
        }

        [Benchmark]
        public void ObjectDetectionYolov8GpuSKBitmap()
        {
            _ = _gpuYolov8.RunObjectDetection(_skBitmap);
        }




        [Benchmark]
        public void ObjectDetectionYolov9CpuSKImage()
        {
            _ = _cpuYolov9.RunObjectDetection(_skImage);
        }

        [Benchmark]
        public void ObjectDetectionYolov9CpuSKBitmap()
        {
            _ = _cpuYolov9.RunObjectDetection(_skBitmap);
        }



        [Benchmark]
        public void ObjectDetectionYolov9GpuSKImage()
        {
            _ = _gpuYolov9.RunObjectDetection(_skImage);
        }

        [Benchmark]
        public void ObjectDetectionYolov9GpuSKBitmap()
        {
            _ = _gpuYolov9.RunObjectDetection(_skBitmap);
        }



        [Benchmark]
        public void ObjectDetectionYolov10CpuSKImage()
        {
            _ = _cpuYolov10.RunObjectDetection(_skImage);
        }

        [Benchmark]
        public void ObjectDetectionYolov10CpuSKBitmap()
        {
            _ = _cpuYolov10.RunObjectDetection(_skBitmap);
        }



        [Benchmark]
        public void ObjectDetectionYolov10GpuSKImage()
        {
            _ = _gpuYolov10.RunObjectDetection(_skImage);
        }

        [Benchmark]
        public void ObjectDetectionYolov10GpuSKBitmap()
        {
            _ = _gpuYolov10.RunObjectDetection(_skBitmap);
        }



        [Benchmark]
        public void ObjectDetectionYolov11CpuSKImage()
        {
            _ = _cpuYolov11.RunObjectDetection(_skImage);
        }

        [Benchmark]
        public void ObjectDetectionYolov11CpuSKBitmap()
        {
            _ = _cpuYolov11.RunObjectDetection(_skBitmap);
        }



        [Benchmark]
        public void ObjectDetectionYolov11GpuSKImage()
        {
            _ = _gpuYolov11.RunObjectDetection(_skImage);
        }

        [Benchmark]
        public void ObjectDetectionYolov11GpuSKBitmap()
        {
            _ = _gpuYolov11.RunObjectDetection(_skBitmap);
        }



        [Benchmark]
        public void ObjectDetectionYolov12CpuSKImage()
        {
            _ = _cpuYolov12.RunObjectDetection(_skImage);
        }

        [Benchmark]
        public void ObjectDetectionYolov12CpuSKBitmap()
        {
            _ = _cpuYolov12.RunObjectDetection(_skBitmap);
        }



        [Benchmark]
        public void ObjectDetectionYolov12GpuSKImage()
        {
            _ = _gpuYolov12.RunObjectDetection(_skImage);
        }

        [Benchmark]
        public void ObjectDetectionYolov12GpuSKBitmap()
        {
            _ = _gpuYolov12.RunObjectDetection(_skBitmap);
        }
    }
}
