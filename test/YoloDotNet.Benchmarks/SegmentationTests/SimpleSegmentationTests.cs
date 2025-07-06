namespace YoloDotNet.Benchmarks.SegmentationTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class SimpleSegmentationTests
    {
        private readonly string _model8 = SharedConfig.GetTestModelV8(ModelType.Segmentation);
        private readonly string _model11 = SharedConfig.GetTestModelV11(ModelType.Segmentation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.People);

        private Yolo _yolo;
        private SKBitmap _image;

        [GlobalSetup]
        public void GlobalSetup()
        {
            _yolo = CreateYolo(YoloParam);
            _image = SKBitmap.Decode(_testImage);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _yolo?.Dispose();
            _image?.Dispose();
        }

        public enum YoloType
        {
            Yolov8_CPU,
            Yolov8_GPU,
            Yolov11_CPU,
            Yolov11_GPU,
        }

        [Params(YoloType.Yolov8_CPU,
            YoloType.Yolov8_GPU,
            YoloType.Yolov11_CPU,
            YoloType.Yolov11_GPU
            )]
        public YoloType YoloParam { get; set; }

        private Yolo CreateYolo(YoloType type)
        {
            return type switch
            {
                YoloType.Yolov8_CPU => new Yolo(new YoloOptions { OnnxModel = _model8, Cuda = false }),
                YoloType.Yolov8_GPU => new Yolo(new YoloOptions { OnnxModel = _model8 }),
                YoloType.Yolov11_CPU => new Yolo(new YoloOptions { OnnxModel = _model11, Cuda = false }),
                YoloType.Yolov11_GPU => new Yolo(new YoloOptions { OnnxModel = _model11 }),
                _ => throw new ArgumentOutOfRangeException()
            };
        }

        [Benchmark]
        public List<Segmentation> Segmentation()
            => _yolo.RunSegmentation(_image);
    }
}
