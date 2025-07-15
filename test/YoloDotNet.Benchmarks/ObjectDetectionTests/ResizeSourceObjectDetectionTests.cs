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
        private SKImage _originalSizeimage;
        private SKImage _modelSizeImage;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            var options = new YoloOptions
            {
                OnnxModel = _model,
                ModelType = ModelType.ObjectDetection,
                Cuda = false
            };

            _cpuYolo = new Yolo(options);

            options.Cuda = true;
            _cudaYolo = new Yolo(options);

            _originalSizeimage = SKImage.FromEncodedData(_originalSizeimagePath);
            _modelSizeImage = SKImage.FromEncodedData(_modelSizeimagePath);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _cudaYolo.Dispose();
            _cpuYolo.Dispose();
            _originalSizeimage.Dispose();
            _modelSizeImage.Dispose();
        }

        [Benchmark]
        public ObjectDetection[] ObjectDetectionOriginalSizeCpu()
        {
            return _cpuYolo.RunObjectDetection(_originalSizeimage);
        }

        [Benchmark]
        public ObjectDetection[] ObjectDetectionOriginalSizeGpu()
        {
            return _cudaYolo.RunObjectDetection(_originalSizeimage);
        }

        [Benchmark]
        public ObjectDetection[] ObjectDetectionModelSizeCpu()
        {
            return _cpuYolo.RunObjectDetection(_modelSizeImage);
        }

        [Benchmark]
        public ObjectDetection[] ObjectDetectionModelSizeGpu()
        {
            return _cudaYolo.RunObjectDetection(_modelSizeImage);
        }

        #endregion Methods
    }
}
