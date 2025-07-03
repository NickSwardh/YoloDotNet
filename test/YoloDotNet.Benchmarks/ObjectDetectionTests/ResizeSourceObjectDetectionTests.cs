namespace YoloDotNet.Benchmarks.ObjectDetectionTests
{
    [MemoryDiagnoser]
    public class ResizeSourceObjectDetectionTests
    {
        #region Fields

        private static readonly string _model = SharedConfig.GetTestModelV8(ModelType.ObjectDetection);
        private static readonly string _originalSizeimagePath = SharedConfig.GetTestImage(ImageType.Street);
        private static readonly string _modelSizeimagePath = SharedConfig.GetTestImage("street640x640.jpg");

        private Yolo _cudaYolo;
        private Yolo _cpuYolo;
        private SKBitmap _originalSizeimage;
        private SKBitmap _modelSizeImage;

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

            options.Cuda = true;
            _cudaYolo = new Yolo(options);

            _originalSizeimage = SKBitmap.Decode(_originalSizeimagePath);
            _modelSizeImage = SKBitmap.Decode(_modelSizeimagePath);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cudaYolo?.Dispose();
            _cpuYolo?.Dispose();
            _originalSizeimage?.Dispose();
            _modelSizeImage?.Dispose();
        }

        [Benchmark]
        public void ObjectDetectionOriginalSizeCpu()
        {
            _ = _cpuYolo.RunObjectDetection(_originalSizeimage);

        }

        [Benchmark]
        public void ObjectDetectionOriginalSizeGpu()
        {
            _ = _cudaYolo.RunObjectDetection(_originalSizeimage);
        }

        [Benchmark]
        public void ObjectDetectionModelSizeCpu()
        {
            _ = _cpuYolo.RunObjectDetection(_modelSizeImage);
        }

        [Benchmark]
        public void ObjectDetectionModelSizeGpu()
        {
            _ = _cudaYolo.RunObjectDetection(_modelSizeImage);
        }

        #endregion Methods
    }
}
