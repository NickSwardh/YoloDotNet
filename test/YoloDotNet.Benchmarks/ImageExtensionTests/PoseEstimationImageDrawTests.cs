namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
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

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            var options = new YoloOptions
            {
                OnnxModel = _model,
                Cuda = false
            };

            _cpuYolo = new Yolo(options);
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

        [Params(true, false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public void DrawPoseEstimationOnSKImage()
        {
            // When drawing using an SKimage, a new SKBitmap is returned with the drawn objects.
            _ = _skImage.Draw(_poseEstimations, CustomKeyPointColorMap.KeyPointOptions, DrawConfidence);
        }

        [Benchmark]
        public void DrawPoseEstimationOnSKBitmap()
        {
            _skBitmap.Draw(_poseEstimations, CustomKeyPointColorMap.KeyPointOptions, DrawConfidence);
        }

        #endregion Methods
    }
}
