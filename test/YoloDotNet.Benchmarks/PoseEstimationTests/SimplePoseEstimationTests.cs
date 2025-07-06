namespace YoloDotNet.Benchmarks.PoseEstimationTests
{
    [MemoryDiagnoser]
    public class SimplePoseEstimationTests
    {
        private readonly string _model8 = SharedConfig.GetTestModelV8(ModelType.PoseEstimation);
        private readonly string _model11 = SharedConfig.GetTestModelV11(ModelType.PoseEstimation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

        private Yolo _yolo;
        private SKBitmap _image;

        // Invoked for each Param parameter
        [GlobalSetup]
        public void GlobalSetup()
        {
            _image = SKBitmap.Decode(_testImage);

            _yolo = CreateYolo(YoloParam);
        }

        // Invoked for each Param parameter
        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _image?.Dispose();
            _yolo?.Dispose();
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
        public List<PoseEstimation> PoseEstimation()
            => _yolo.RunPoseEstimation(_image);
    }
}
