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

        private SKBitmap _skBitmap;
        private SKImage _skImage;
        private Yolo _yolo;
        private ImageResize _samplingOptions;

        // Invoked for each Param parameter
        [GlobalSetup]
        public void GlobalSetup()
        {
            _skBitmap = SKBitmap.Decode(_testImage);
            _skImage = SKImage.FromEncodedData(_testImage);
            _samplingOptions = ImageResize.Proportional;

            _yolo = CreateYolo(YoloParam);
        }

        // Invoked for each Param parameter
        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _skImage?.Dispose();
            _skBitmap?.Dispose();
            _yolo.Dispose();
        }

        public enum YoloType
        {
            Yolov5u_CPU,
            Yolov5u_GPU,
            Yolov8_CPU,
            Yolov8_GPU,
            Yolov9_CPU,
            Yolov9_GPU,
            Yolov10_CPU,
            Yolov10_GPU,
            Yolov11_CPU,
            Yolov11_GPU,
            Yolov12_CPU,
            Yolov12_GPU,
        }

        [Params(YoloType.Yolov5u_CPU,
            YoloType.Yolov5u_GPU,
            YoloType.Yolov8_CPU,
            YoloType.Yolov8_GPU,
            YoloType.Yolov9_CPU,
            YoloType.Yolov9_GPU,
            YoloType.Yolov10_CPU,
            YoloType.Yolov10_GPU,
            YoloType.Yolov11_CPU,
            YoloType.Yolov11_GPU,
            YoloType.Yolov12_CPU,
            YoloType.Yolov12_GPU
            )]
        public YoloType YoloParam { get; set; }

        private Yolo CreateYolo(YoloType type)
        {
            return type switch
            {
                YoloType.Yolov5u_CPU => new Yolo(new YoloOptions { OnnxModel = _modelV5U, ImageResize = _samplingOptions, Cuda = false }),
                YoloType.Yolov5u_GPU => new Yolo(new YoloOptions { OnnxModel = _modelV5U, ImageResize = _samplingOptions }),
                YoloType.Yolov8_CPU => new Yolo(new YoloOptions { OnnxModel = _modelV8, ImageResize = _samplingOptions, Cuda = false }),
                YoloType.Yolov8_GPU => new Yolo(new YoloOptions { OnnxModel = _modelV8, ImageResize = _samplingOptions }),
                YoloType.Yolov9_CPU => new Yolo(new YoloOptions { OnnxModel = _modelV9, ImageResize = _samplingOptions, Cuda = false }),
                YoloType.Yolov9_GPU => new Yolo(new YoloOptions { OnnxModel = _modelV9, ImageResize = _samplingOptions }),
                YoloType.Yolov10_CPU => new Yolo(new YoloOptions { OnnxModel = _modelV10, ImageResize = _samplingOptions, Cuda = false }),
                YoloType.Yolov10_GPU => new Yolo(new YoloOptions { OnnxModel = _modelV10, ImageResize = _samplingOptions }),
                YoloType.Yolov11_CPU => new Yolo(new YoloOptions { OnnxModel = _modelV11, ImageResize = _samplingOptions, Cuda = false }),
                YoloType.Yolov11_GPU => new Yolo(new YoloOptions { OnnxModel = _modelV11, ImageResize = _samplingOptions }),
                YoloType.Yolov12_CPU => new Yolo(new YoloOptions { OnnxModel = _modelV12, ImageResize = _samplingOptions, Cuda = false }),
                YoloType.Yolov12_GPU => new Yolo(new YoloOptions { OnnxModel = _modelV12, ImageResize = _samplingOptions }),
                _ => throw new ArgumentOutOfRangeException()
            };
        }

        [Benchmark]
        public void ObjectDetection()
            => _yolo.RunObjectDetection(_skBitmap);
    }
}
