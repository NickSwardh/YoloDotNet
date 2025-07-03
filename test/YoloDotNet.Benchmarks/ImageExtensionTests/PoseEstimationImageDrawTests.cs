namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class PoseEstimationImageDrawTests
    {
        #region Fields

        private readonly string _model = SharedConfig.GetTestModelV8(ModelType.PoseEstimation);
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

        private Yolo _cpuYolo;
        private SKImage _skImage;
        private SKBitmap _skBitmap;
        private List<PoseEstimation> _poseEstimations;
        private PoseDrawingOptions _poseDrawingOptions;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cpuYolo = new Yolo(new YoloOptions
            {
                OnnxModel = _model,
                Cuda = false
            });

            _poseDrawingOptions = new PoseDrawingOptions
            {
                KeyPointMarkers = CustomKeyPointColorMap.KeyPoints
            };

            _skBitmap = SKBitmap.Decode(_testImage);
            _skImage = SKImage.FromEncodedData(_testImage);

            // We just need one result to use for drawing.
            _poseEstimations = _cpuYolo.RunPoseEstimation(_skImage);
        }

        [GlobalCleanup]
        public void CleanUp()
        {
            _cpuYolo?.Dispose();
            _skImage?.Dispose();
            _skBitmap?.Dispose();
        }

        [Benchmark]
        public void DrawPoseEstimationOnSKImage()
        {
            // When drawing using an SKimage, a new SKBitmap is returned with the drawn objects.
            _ = _skImage.Draw(_poseEstimations, _poseDrawingOptions);
        }

        [Benchmark]
        public void DrawPoseEstimationOnSKBitmap()
        {
            _skBitmap.Draw(_poseEstimations, _poseDrawingOptions);
        }

        #endregion Methods
    }
}
