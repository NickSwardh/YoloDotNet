namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class NormalizePixelsToTensorTests
    {
        #region Fields

        private static readonly string _model = SharedConfig.GetTestModelV8(ModelType.ObjectDetection);
        private static readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);
        private ArrayPool<float> _customSizeFloatPool;

        private int _tensorBufferSize;

        private Yolo _cpuYolo;

        private SKBitmap _image;
        private float[] _tensorArrayBuffer;

        private static SKBitmap _resizedBitmap;

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
            _image = SKBitmap.Decode(_testImage);

            var imageInfo = new SKImageInfo(_cpuYolo.OnnxModel.Input.Width, _cpuYolo.OnnxModel.Input.Height, SKColorType.Rgb888x, SKAlphaType.Opaque);

            _resizedBitmap = _image.ResizeImageProportional(imageInfo, options.SamplingOptions);

            _tensorBufferSize = _cpuYolo.OnnxModel.Input.BatchSize * _cpuYolo.OnnxModel.Input.Channels * _cpuYolo.OnnxModel.Input.Width * _cpuYolo.OnnxModel.Input.Height;
            _customSizeFloatPool = ArrayPool<float>.Create(_tensorBufferSize + 1, 10);
            _tensorArrayBuffer = _customSizeFloatPool.Rent(_tensorBufferSize);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _customSizeFloatPool.Return(_tensorArrayBuffer, true);
            _resizedBitmap?.Dispose();
            _image?.Dispose();
            _cpuYolo?.Dispose();
        }

        [Benchmark]
        public void NormalizePixelsToTensor()
        {
            _ = _resizedBitmap.NormalizePixelsToTensor(_cpuYolo.OnnxModel.InputShape, _tensorBufferSize, _tensorArrayBuffer);
        }

        #endregion Methods
    }
}
